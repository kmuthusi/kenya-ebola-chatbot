from pathlib import Path

import pytest

import simple_rag
import model_warmup
from conversation import clarification_for, contextualize_question
from public_safety import requires_professional_care
from bounded_composer import REFUSAL
from rag_pipeline import (
    exclude_bibliography,
    is_general_management_question,
    is_overview_question,
    management_evidence,
    overview_evidence,
)
from retrievers import GuidelineRetriever, _read_pdf
from simple_rag import SimpleAssistantRAG, SimpleAssistantRequest
from verifier import verify_answer


def test_pdf_is_extractable_and_page_numbers_are_pdf_positions():
    pdf = next(Path(__file__).resolve().parents[1].glob("*.pdf"))
    pages = GuidelineRetriever(str(pdf)).read_and_chunk()
    assert len(pages) > 50
    assert pages[0]["page"] == 1
    assert any("Isolation" in item["text"] for item in pages)


def test_warmup_loads_embedding_then_chat_model(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_warmup.ollama, "embed", lambda **kwargs: calls.append(("embed", kwargs))
    )
    monkeypatch.setattr(
        model_warmup.ollama, "generate", lambda **kwargs: calls.append(("generate", kwargs))
    )
    model_warmup.warm_models("llama3.1:8b", "nomic-embed-text:latest")
    assert [name for name, _ in calls] == ["embed", "generate"]
    assert calls[1][1]["prompt"] == ""
    assert all(args["keep_alive"] == model_warmup.KEEP_ALIVE for _, args in calls)


@pytest.mark.parametrize(
    "question",
    [
        "How should I manage an Ebola patient?",
        "How to administer mAb114?",
        "What dose of medicine should I use?",
        "What does the guideline say about IV fluid dosing?",
        "My child has fever after travel.",
        "I have fever and think I have Ebola.",
    ],
)
def test_public_care_requests_are_redirected(question):
    assert requires_professional_care(question)


def test_general_document_question_is_available_to_public():
    assert not requires_professional_care("What topics does the guideline cover?")


def test_follow_up_uses_only_recent_user_questions_for_context():
    history = ("What is Ebola?", "How does it spread?")
    assert contextualize_question("How is it transmitted?", history) == "How is Ebola transmitted?"
    assert "Which aspect of Ebola in children" in clarification_for("What about children?", history)
    assert contextualize_question("What about Marburg?", history) == (
        "What does the guideline say about Marburg disease?"
    )
    assert "compare Marburg with Ebola" in clarification_for("What about Marburg?", history)
    assert contextualize_question("symptoms", ("What is Ebola?", "What about children?")) == (
        "What does the guideline say about symptoms of Ebola in children?"
    )
    assert contextualize_question("comparison", ("What is Ebola?", "What about Marburg?")) == (
        "How does the guideline compare Marburg with Ebola?"
    )
    assert contextualize_question("What is Marburg?", history) == "What is Marburg?"


def test_ambiguous_follow_up_asks_before_retrieval(monkeypatch):
    class FakeStore:
        def __init__(self, **kwargs):
            pass

        def ensure_index(self, pages):
            pytest.fail("Ambiguous follow-up should not reach retrieval")

    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    rag = SimpleAssistantRAG("guideline.pdf")
    result = rag.ask(SimpleAssistantRequest(
        "What about children?", earlier_questions=("What is Ebola?",)
    ))
    assert "Which aspect of Ebola in children" in result["answer"]
    assert result["chunks"] == []
    switched = rag.ask(SimpleAssistantRequest(
        "What about Marburg?", earlier_questions=("What is Ebola?",)
    ))
    assert "compare Marburg with Ebola" in switched["answer"]
    assert switched["chunks"] == []


def test_bibliography_is_excluded_from_ordinary_answers():
    pages = [
        {"page": 12, "text": "Ebola disease overview."},
        {"page": 78, "text": "71\n11. References\nWHO guideline."},
    ]
    candidates = [
        {"page": 78, "text": "WHO guideline.", "score": 0.8},
        {"page": 12, "text": "Ebola disease overview.", "score": 0.7},
    ]
    assert [item["page"] for item in exclude_bibliography("What is Ebola?", candidates, pages)] == [12]
    assert exclude_bibliography("Show references", candidates, pages) == candidates


def test_pdf_extraction_is_reused_and_general_care_uses_care_chapter():
    pdf = next(Path(__file__).resolve().parents[1].glob("*.pdf"))
    _read_pdf.cache_clear()
    retriever = GuidelineRetriever(str(pdf))
    pages = retriever.read_and_chunk()
    retriever.read_and_chunk()
    assert _read_pdf.cache_info().hits == 1
    assert is_general_management_question("how should I manage an ebola patient?")
    assert not is_general_management_question("how should I manage shock in an Ebola patient?")
    assert [item["page"] for item in management_evidence(pages)] == [22, 21, 30]


def test_verifier_withholds_uncited_and_wrongly_cited_answers():
    evidence = [{"page": 21, "text": "Isolation guidance."}]
    assert verify_answer("Use isolation (PDF Page 21).", evidence)["issues"] == []
    assert verify_answer("Use isolation.", evidence)["verified_answer"] == REFUSAL
    assert verify_answer("Use isolation (PDF Page 99).", evidence)["verified_answer"] == REFUSAL
    assert verify_answer("- Isolate (PDF Page 21).\n- Monitor symptoms.", evidence)["verified_answer"] == REFUSAL


def test_orchestrator_uses_guideline_evidence(monkeypatch):
    class FakeStore:
        def __init__(self, **kwargs):
            self.indexed = False

        def ensure_index(self, pages):
            self.indexed = bool(pages)

        def query(self, question, k, include_scores):
            assert self.indexed
            return [{"page": 21, "text": "Suspected cases should be isolated.", "score": 0.8}]

    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(GuidelineRetriever, "read_and_chunk", lambda self: [{"page": 21, "text": "Suspected cases should be isolated."}])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda question, evidence, model: "Suspected cases should be isolated (PDF Page 21).")
    rag = SimpleAssistantRAG("guideline.pdf")
    result = rag.ask(SimpleAssistantRequest("How are suspected cases isolated?"))
    assert result["issues"] == []
    assert result["chunks"][0]["page"] == 21


def test_empty_question_is_rejected(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", lambda **kwargs: object())
    rag = SimpleAssistantRAG("guideline.pdf")
    with pytest.raises(ValueError):
        rag.ask(SimpleAssistantRequest(" "))


def test_overview_uses_document_scope_not_table_of_contents(monkeypatch):
    pages = [
        {"page": 4, "text": "These guidelines provide screening and care. They are for health workers."},
        {"page": 5, "text": "1.1 Purpose ........ 5\n1.2 Objectives ........ 5"},
        {"page": 10, "text": "These Case Management Guidelines cover care and IPC. The Constitution"},
        {"page": 12, "text": "This guideline focuses specifically on Ebola. Ebola virus disease.\n"
         "1.1 Purpose\nTo provide practical guidance.\n1.2 Objectives"},
    ]
    evidence = overview_evidence(pages)
    assert [item["page"] for item in evidence] == [4, 10, 12, 12]
    assert is_overview_question("tell me about this guidelines")
    assert is_overview_question("What does the guideline cover?")
    assert not is_overview_question("How should a suspected case be isolated under the guideline?")
    assert not is_overview_question("Summarize isolation guidance in this guideline")

    class FakeStore:
        def __init__(self, **kwargs):
            pass

        def ensure_index(self, pages):
            pass

        def query(self, *args, **kwargs):
            pytest.fail("Overview should use the document's scope and purpose")

    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(GuidelineRetriever, "read_and_chunk", lambda self: pages)
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "The guideline covers care.")
    monkeypatch.setattr(
        simple_rag, "repair_citations",
        lambda question, draft, evidence, model: "The guideline covers care (PDF Page 10).",
    )
    result = SimpleAssistantRAG("guideline.pdf").ask(
        SimpleAssistantRequest("tell me about this guidelines")
    )
    assert result["answer"] == "The guideline covers care (PDF Page 10)."
    assert result["issues"] == []
