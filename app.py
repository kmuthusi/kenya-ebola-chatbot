"""Streamlit interface for the Kenya Ebola Care Guide."""

import os
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Thread

import streamlit as st

from model_warmup import warm_models
from public_safety import CARE_REDIRECT, requires_professional_care
from retrievers import DEFAULT_PDF, GuidelineRetriever
from simple_rag import DEFAULT_MODEL, SimpleAssistantRAG, SimpleAssistantRequest
from vector_store import DEFAULT_EMBED_MODEL


LOGGER = logging.getLogger(__name__)


@st.cache_resource
def answer_slots() -> BoundedSemaphore:
    return BoundedSemaphore(max(1, int(os.getenv("MAX_CONCURRENT_ANSWERS", "2"))))


@st.cache_resource(show_spinner=False, ttl="25m")
def start_model_warmup(chat_model: str, embed_model: str, pdf_path: str) -> Future:
    """Prepare models and PDF without delaying the app page."""
    future = Future()

    def load() -> None:
        try:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                models = pool.submit(warm_models, chat_model, embed_model)
                pages = pool.submit(GuidelineRetriever(pdf_path).read_and_chunk)
                models.result()
                pages.result()
            future.set_result(round(time.perf_counter() - started, 2))
        except Exception as exc:
            LOGGER.exception("Guideline assistant startup failed")
            future.set_exception(exc)

    Thread(target=load, daemon=True, name="ollama-model-warmup").start()
    return future


st.set_page_config(page_title="Kenya Ebola Care Guide")
st.markdown(
    """
    <style>
    @media (max-width: 600px) {
      [data-testid="stMainBlockContainer"] {
        padding: 1.25rem 1rem 2rem !important;
        max-width: 100% !important;
      }
      [data-testid="stMainBlockContainer"] h1 {
        font-size: 1.9rem;
        line-height: 1.15;
      }
      [data-testid="stChatInput"] textarea {
        font-size: 16px;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Kenya Ebola Care Guide")
st.info(
    "Based on Kenya's October 2026 *Case Management Guidelines for Filovirus Disease*. "
    "Ask about Ebola and related filovirus care; answers cite PDF pages."
)
st.caption(
    "General information only. For suspected infection, contact a health facility or "
    "public health team. Do not enter personal details."
)

pdf_path = os.getenv("GUIDELINE_PDF_PATH", DEFAULT_PDF)
model = DEFAULT_MODEL
top_k = max(1, min(10, int(os.getenv("EVIDENCE_PASSAGES", "5"))))
embed_model = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
warmup = start_model_warmup(model, embed_model, pdf_path)


@st.fragment(run_every="5s")
def show_warmup_status() -> None:
    if not warmup.done():
        st.caption("Preparing the guideline assistant...")
        return
    try:
        warmup.result()
    except Exception:
        st.warning("The guideline assistant is temporarily unavailable.")
        if st.button("Try again"):
            start_model_warmup.clear()
            st.rerun()


show_warmup_status()

def show_sources(chunks: list[dict]) -> None:
    if not chunks:
        return
    by_page: dict[int, list[str]] = {}
    for item in chunks:
        texts = by_page.setdefault(item["page"], [])
        if item["text"] not in texts:
            texts.append(item["text"])
    with st.expander("View source passages"):
        for page, texts in by_page.items():
            st.markdown(f"**PDF Page {page}**")
            for text in texts:
                st.write(text)


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages and st.button("New chat", type="tertiary"):
    st.session_state.messages = []
    st.rerun()

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hello! Ask about the guideline's scope, screening, isolation, or supportive care.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("issues"):
            st.warning("This answer could not be verified against the guideline.")
        show_sources(message.get("chunks", []))

if question := st.chat_input("Ask about the guideline", max_chars=500):
    earlier_questions = tuple(
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    )[-2:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if requires_professional_care(question):
            result = {"answer": CARE_REDIRECT, "chunks": [], "issues": []}
        else:
            slots = answer_slots()
            if not slots.acquire(blocking=False):
                result = {"answer": "The assistant is busy. Please try again shortly.",
                          "chunks": [], "issues": []}
            else:
                try:
                    with st.spinner("Checking the guideline..."):
                        warmup.result(timeout=180)
                        result = SimpleAssistantRAG(pdf_path, model).ask(
                            SimpleAssistantRequest(question, top_k, model, earlier_questions)
                        )
                except Exception:
                    LOGGER.exception("Guideline question failed")
                    result = {"answer": "The guideline assistant is temporarily unavailable. "
                              "Please try again later.", "chunks": [], "issues": []}
                finally:
                    slots.release()
        st.markdown(result["answer"])
        if result["issues"]:
            st.warning("This answer could not be verified against the guideline.")
        show_sources(result["chunks"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"],
         "chunks": result["chunks"], "issues": result["issues"]}
    )
    st.session_state.messages = st.session_state.messages[-20:]
