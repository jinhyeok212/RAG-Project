"""Unit tests for read-only retrieval result normalization."""

import pytest

from src.rag.retriever import RetrievalConfig, Retriever


class FakeEmbedder:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self):
        self.query_kwargs = None

    def count(self):
        return 2

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["first text", "second text"]],
            "metadatas": [[
                {"document_id": "doc-1", "section": "overview"},
                {"document_id": "doc-2", "section": "eligibility"},
            ]],
            "distances": [[0.1, 0.25]],
        }


def test_retrieve_normalizes_chroma_result():
    collection = FakeCollection()
    result = Retriever(FakeEmbedder(), collection).retrieve("q-1", "청년 지원은?", 5)

    assert result["query_id"] == "q-1"
    assert result["evaluation_max_k"] == 5
    assert result["index_version"] == "index_baseline_kure_v1"
    assert collection.query_kwargs == {
        "query_embeddings": [[0.1, 0.2, 0.3]],
        "n_results": 2,
        "include": ["documents", "metadatas", "distances"],
    }
    assert result["results"][0]["rank"] == 1
    assert result["results"][0]["document_id"] == "doc-1"
    assert result["results"][0]["distance"] == 0.1
    assert result["results"][0]["similarity"] == 0.9


@pytest.mark.parametrize(
    ("query_id", "query", "top_k"),
    [("", "question", 5), ("q-1", "   ", 5), ("q-1", "question", 0)],
)
def test_retrieve_rejects_invalid_input(query_id, query, top_k):
    with pytest.raises(ValueError):
        Retriever(FakeEmbedder(), FakeCollection()).retrieve(query_id, query, top_k)


def test_config_rejects_non_cosine_distance():
    with pytest.raises(ValueError):
        RetrievalConfig(distance_metric="l2")


def test_retrieve_rejects_inconsistent_chroma_response():
    collection = FakeCollection()
    collection.query = lambda **_: {
        "ids": [["chunk-1"]], "documents": [[]],
        "metadatas": [[{"document_id": "doc-1"}]], "distances": [[0.1]],
    }
    with pytest.raises(ValueError, match="inconsistent lengths"):
        Retriever(FakeEmbedder(), collection).retrieve("q-1", "question")
