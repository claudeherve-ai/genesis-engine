"""Tests for the SQLite-backed knowledge store + deterministic TF-IDF search."""

import pytest

from genesis.knowledge import KnowledgeStore, KnowledgeDocument, tokenize


def test_tokenize_drops_stopwords_and_short_tokens():
    toks = tokenize("The quick a brown FOX is on it")
    assert "the" not in toks
    assert "is" not in toks
    assert "a" not in toks  # stopword + short
    assert "quick" in toks
    assert "brown" in toks
    assert "fox" in toks


def test_ingest_and_count(db_session):
    store = KnowledgeStore(db_session)
    assert store.count() == 0
    doc_id = store.ingest("Title", "some meaningful body text", source="unit")
    assert isinstance(doc_id, str) and doc_id
    assert store.count() == 1


def test_ingest_rejects_empty_text(db_session):
    store = KnowledgeStore(db_session)
    with pytest.raises(ValueError):
        store.ingest("Title", "   ")


def test_clear_removes_all(db_session):
    store = KnowledgeStore(db_session)
    store.ingest("A", "alpha bravo charlie")
    store.ingest("B", "delta echo foxtrot")
    assert store.count() == 2
    removed = store.clear()
    assert removed == 2
    assert store.count() == 0


def test_search_ranks_relevant_first(db_session):
    store = KnowledgeStore(db_session)
    store.ingest("Kubernetes", "kubernetes pods deployments and services scaling")
    store.ingest("Cooking", "recipes for pasta tomato sauce and basil")

    results = store.search("kubernetes scaling pods", k=5)
    assert results
    top_doc, top_score = results[0]
    assert "Kubernetes" == top_doc.title
    assert top_score > 0


def test_search_empty_query_returns_empty(db_session):
    store = KnowledgeStore(db_session)
    store.ingest("Doc", "content here about things")
    assert store.search("") == []
    # Stop-word-only query also yields nothing.
    assert store.search("the is a of") == []


def test_search_zero_overlap_excluded(db_session):
    store = KnowledgeStore(db_session)
    store.ingest("Doc", "alpha bravo charlie")
    assert store.search("zzz nonexistent terms") == []


def test_search_single_doc_corpus_has_signal(db_session):
    # Smoothed IDF must still rank a single document above zero.
    store = KnowledgeStore(db_session)
    store.ingest("Only", "databricks lakehouse unity catalog governance")
    results = store.search("databricks governance", k=3)
    assert len(results) == 1
    assert results[0][1] > 0


def test_search_tie_break_is_deterministic(db_session):
    # Two identical documents → identical score → stable id-asc ordering.
    store = KnowledgeStore(db_session)
    id1 = store.ingest("Same", "identical token payload here")
    id2 = store.ingest("Same", "identical token payload here")
    results = store.search("identical token payload", k=5)
    assert len(results) == 2
    returned_ids = [doc.id for doc, _ in results]
    assert returned_ids == sorted([id1, id2])


def test_search_respects_k(db_session):
    store = KnowledgeStore(db_session)
    for i in range(5):
        store.ingest(f"Doc{i}", "shared keyword payload alpha beta")
    results = store.search("shared keyword payload", k=2)
    assert len(results) == 2


def test_all_documents_round_trip(db_session):
    store = KnowledgeStore(db_session)
    store.ingest("T", "body", source="src", metadata={"k": "v"})
    docs = store.all_documents()
    assert len(docs) == 1
    assert isinstance(docs[0], KnowledgeDocument)
    assert docs[0].source == "src"
    assert docs[0].metadata == {"k": "v"}
