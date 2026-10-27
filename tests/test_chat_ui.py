from pathlib import Path

from streamlit.testing.v1 import AppTest

import model_warmup
import simple_rag
from retrievers import GuidelineRetriever


def test_chat_keeps_turns_and_can_start_over(monkeypatch):
    seen = []

    class FakeRAG:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self, request):
            seen.append((request.question, request.earlier_questions))
            return {"answer": "Guideline answer (PDF Page 12).", "issues": [],
                    "chunks": [{"page": 12, "text": "Guideline evidence."}]}

    monkeypatch.setattr(model_warmup, "warm_models", lambda *args: 0.0)
    monkeypatch.setattr(GuidelineRetriever, "read_and_chunk", lambda self: [{"page": 12, "text": "Guideline evidence."}])
    monkeypatch.setattr(simple_rag, "SimpleAssistantRAG", FakeRAG)

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=10)
    app.chat_input[0].set_value("What is Ebola?")
    app.run(timeout=10)
    app.chat_input[0].set_value("How is it transmitted?")
    app.run(timeout=10)

    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant", "user", "assistant"]
    assert seen == [
        ("What is Ebola?", ()),
        ("How is it transmitted?", ("What is Ebola?",)),
    ]
    assert len(app.session_state.messages) == 4
    app.button[0].click()
    app.run(timeout=10)
    assert app.session_state.messages == []
