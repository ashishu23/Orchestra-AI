def rrf_combine(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Each item must have an 'id' key for deduplication.
    Score formula: sum(1 / (k + rank_i)) across all lists.

    Args:
        ranked_lists: List of ranked result lists; each item is a dict with 'id'.
        k: RRF constant (default 60, higher = less penalty for lower ranks).

    Returns:
        Single merged list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            items[doc_id] = item

    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    result = []
    for doc_id in sorted_ids:
        entry = dict(items[doc_id])
        entry["rrf_score"] = scores[doc_id]
        result.append(entry)
    return result
