from src.data.privacy_masking import mask_text
from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from src.retrieval.rrf import reciprocal_rank_fusion


def test_privacy_masking_phone_email_order():
    text = "010-1234-5678 test@example.com 주문번호 AB12345"
    masked = mask_text(text)
    assert "010-1234-5678" not in masked
    assert "test@example.com" not in masked
    assert "[PHONE]" in masked
    assert "[EMAIL]" in masked


def test_rrf_merges_duplicate_doc_ids():
    rows = reciprocal_rank_fusion(
        [
            [{"doc_id": "a", "bm25_rank": 1, "bm25_score": 10.0}],
            [{"doc_id": "a", "dense_rank": 1, "dense_score": 0.9}, {"doc_id": "b", "dense_rank": 2, "dense_score": 0.8}],
        ]
    )
    assert rows[0]["doc_id"] == "a"
    assert rows[0]["bm25_rank"] == 1
    assert rows[0]["dense_rank"] == 1


def test_retrieval_metrics():
    results = ["x", "a", "b"]
    positives = {"a", "c"}
    assert hit_at_k(results, positives, 2) == 1.0
    assert recall_at_k(results, positives, 3) == 0.5
    assert mrr(results, positives) == 0.5
    assert ndcg_at_k(results, positives, 3) > 0
