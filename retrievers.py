"""Extract guideline text with PDF page numbers for citations."""

from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader


DEFAULT_PDF = "Final Case Management Guidelines for Ebola Virus- 23102026 V5.pdf"


@lru_cache(maxsize=4)
def _read_pdf(path: Path, mtime_ns: int, size: int) -> tuple[tuple[int, str], ...]:
    """Reuse extracted pages until the PDF's timestamp or size changes."""
    reader = PdfReader(str(path))
    return tuple(
        (number, text.strip())
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    )


class GuidelineRetriever:
    def __init__(self, pdf_path: str = DEFAULT_PDF):
        self.pdf_path = Path(pdf_path)

    def read_and_chunk(self) -> list[dict]:
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"Guideline PDF not found: {self.pdf_path}")
        path = self.pdf_path.resolve()
        stat = path.stat()
        pages = [
            {"page": number, "text": text, "source": path.name}
            for number, text in _read_pdf(path, stat.st_mtime_ns, stat.st_size)
        ]
        if not pages:
            raise ValueError("The guideline PDF has no extractable text.")
        return pages
