# tests/test_pipeline.py

import pytest
from fastapi.testclient import TestClient
from app.retriever import retrieve, load_vectorstore
from app.chain import ask
from main import app

# ── FastAPI test client ────────────────────────────────────────
# TestClient lets you call FastAPI endpoints without running a server
client = TestClient(app)


# ══════════════════════════════════════════════════════════════
# RETRIEVER TESTS
# ══════════════════════════════════════════════════════════════

def test_retriever_returns_results():
    """Retriever should return chunks for any valid query."""
    results = retrieve("What is privacy?")
    assert len(results) > 0


def test_retriever_returns_correct_type():
    """Each result should be a LangChain Document with page_content."""
    results = retrieve("freedom of speech")
    for doc in results:
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")
        assert isinstance(doc.page_content, str)
        assert len(doc.page_content) > 0


def test_retriever_topk():
    """Retriever should respect the k parameter."""
    results = retrieve("privacy", k=2)
    assert len(results) <= 2


def test_retriever_relevance():
    """Privacy query should return chunks containing privacy-related content."""
    results = retrieve("What does the constitution say about privacy?")
    combined = " ".join([doc.page_content for doc in results]).lower()
    assert "privacy" in combined or "article 10" in combined


# ══════════════════════════════════════════════════════════════
# CHAIN TESTS
# ══════════════════════════════════════════════════════════════

def test_chain_returns_string():
    """ask() should always return a non-empty string."""
    answer = ask("What does the constitution say about privacy?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_chain_cites_article():
    """Chain should return a grounded answer or an honest 'not found'."""
    answer = ask("What does Article 10 say?")
    found_content = "article" in answer.lower() or "privacy" in answer.lower()
    honest_fallback = "cannot find" in answer.lower()
    assert found_content or honest_fallback


def test_chain_handles_unknown_question():
    """Pipeline should gracefully handle questions outside the document."""
    answer = ask("What is the capital of France?")
    assert isinstance(answer, str)
    # Should either say it can't find it, or give a grounded response
    assert len(answer) > 0


# ══════════════════════════════════════════════════════════════
# API TESTS
# ══════════════════════════════════════════════════════════════

def test_api_root():
    """Root endpoint should return 200 with status ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_health():
    """Health endpoint should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_ask_valid_question():
    """Valid question should return 200 with question and answer fields."""
    response = client.post(
        "/ask",
        json={"question": "What does the constitution say about privacy?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_api_ask_empty_question():
    """Empty question should return 400 Bad Request."""
    response = client.post(
        "/ask",
        json={"question": "   "}
    )
    assert response.status_code == 400


def test_api_ask_missing_field():
    """Request with missing question field should return 422 Unprocessable Entity."""
    response = client.post(
        "/ask",
        json={}
    )
    assert response.status_code == 422


def test_api_ask_wrong_type():
    """Request with wrong type for question should return 422."""
    response = client.post(
        "/ask",
        json={"question": 12345}
    )
    assert response.status_code == 422