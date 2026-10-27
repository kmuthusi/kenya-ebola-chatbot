"""Ollama embeddings, Chroma persistence, and freshness-aware querying."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
EMBED_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


def _hash_manifest(items: List[Dict[str, Any]]) -> str:
    """Hash source content, not just page lengths, so edits trigger reindexing."""
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item.get("page")).encode())
        digest.update(str(item.get("text", "")).encode())
    return digest.hexdigest()


class VectorStore:
    def __init__(
        self,
        collection_name: str = "kenya_ebola_guideline_ollama",
        chroma_dir: str = "chroma_db",
        embed_model: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model or os.getenv(
            "OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL
        )
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.manifest_path = self.chroma_dir / f"{collection_name}.manifest.json"
        self._manifest_hash: Optional[str] = None
        self._manifest_model: Optional[str] = None
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text())
                self._manifest_hash = manifest.get("hash")
                self._manifest_model = manifest.get("embed_model")
            except (OSError, json.JSONDecodeError):
                self._manifest_hash = None

    def _chunk_text(self, text: str, chunk_len: int = 1200, overlap: int = 200) -> List[str]:
        if chunk_len <= 0 or overlap < 0 or overlap >= chunk_len:
            raise ValueError("chunk_len must be positive and overlap must be smaller than chunk_len")
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_len, length)
            chunks.append(text[start:end])
            if end == length:
                break
            start = end - overlap
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 40]

    def index_chunks(self, raw_chunks: List[Dict[str, Any]]) -> None:
        if not raw_chunks:
            raise ValueError("Cannot index an empty source.")
        manifest = _hash_manifest(raw_chunks)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)

        batch_size = 32
        records = []
        for source_index, raw in enumerate(raw_chunks):
            for sub_index, text in enumerate(self._chunk_text(raw["text"])):
                records.append((source_index, sub_index, text, raw.get("page")))

        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            embeddings = ollama.embed(
                model=self.embed_model,
                input=[record[2] for record in batch],
                keep_alive=EMBED_KEEP_ALIVE,
            )["embeddings"]
            self.collection.add(
                ids=[
                    f"chunk-{source}-{sub}-{hashlib.sha1(text.encode()).hexdigest()[:12]}"
                    for source, sub, text, _ in batch
                ],
                documents=[record[2] for record in batch],
                metadatas=[{"page": record[3]} for record in batch],
                embeddings=embeddings,
            )

        self._manifest_hash = manifest
        self._manifest_model = self.embed_model
        self.manifest_path.write_text(
            json.dumps({"hash": manifest, "embed_model": self.embed_model}, indent=2)
        )

    def ensure_index(self, raw_chunks: List[Dict[str, Any]]) -> None:
        manifest = _hash_manifest(raw_chunks)
        if self._manifest_hash != manifest or self._manifest_model != self.embed_model or self.collection.count() == 0:
            self.index_chunks(raw_chunks)

    def query(self, query: str, k: int = 5, include_scores: bool = True) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        query_embedding = ollama.embed(
            model=self.embed_model, input=query, keep_alive=EMBED_KEEP_ALIVE
        )["embeddings"][0]
        include = ["documents", "metadatas", "distances"] if include_scores else ["documents", "metadatas"]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=include,
        )
        documents = result.get("documents", [[]])[0]
        metadata = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if include_scores else [None] * len(documents)
        output = []
        for document, meta, distance in zip(documents, metadata, distances):
            output.append(
                {
                    "text": document,
                    "page": meta.get("page"),
                    "score": (1.0 / (1.0 + distance)) if distance is not None else None,
                }
            )
        return output
