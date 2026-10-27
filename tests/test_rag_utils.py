from src.rag import split_text, classify_query


def test_split_text_returns_chunks():
    chunks = split_text("a" * 2500, chunk_size=1000, overlap=150)
    assert len(chunks) >= 3
    assert all(chunks)


def test_empty_text_has_no_chunks():
    assert split_text("   ") == []


def test_router_detects_document_questions():
    assert classify_query("According to my document, what is RAG?") == "index"


def test_router_detects_current_information():
    assert classify_query("What are the latest AI developments in 2026?") == "search"


def test_router_defaults_to_general():
    assert classify_query("Explain recursion in Python") == "general"
