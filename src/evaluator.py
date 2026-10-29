
from __future__ import annotations

from .data_loader import snippets_for


def char_metrics(results: list[dict], gold_spans: list[dict], k: int) -> dict:
    """Measure character-level precision, recall, and F1 at k."""

    retrieved = results[:k]

    # Store character positions as intervals grouped by document.
    predicted = {}
    gold = {}

    for result in retrieved:
        doc_id = result["doc_id"].replace("\\", "/")
        predicted.setdefault(doc_id, []).append(
            (int(result["start"]), int(result["end"]))
        )

    for span in gold_spans:
        doc_id = span["file_path"].replace("\\", "/")
        start, end = int(span["start"]), int(span["end"])

        if start >= 0 and end >= start:
            gold.setdefault(doc_id, []).append((start, end))

    def merge(intervals):
        merged = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged

    predicted = {
        doc: merge(intervals)
        for doc, intervals in predicted.items()
    }
    gold = {
        doc: merge(intervals)
        for doc, intervals in gold.items()
    }

    def total_length(intervals):
        return sum(end - start for spans in intervals.values()
                   for start, end in spans)

    overlap = 0

    for doc_id, pred_intervals in predicted.items():
        gold_intervals = gold.get(doc_id, [])

        for p_start, p_end in pred_intervals:
            for g_start, g_end in gold_intervals:
                overlap += max(
                    0,
                    min(p_end, g_end) - max(p_start, g_start)
                )

    predicted_chars = total_length(predicted)
    gold_chars = total_length(gold)

    precision = (
        overlap / predicted_chars if predicted_chars else 0.0
    )
    recall = overlap / gold_chars if gold_chars else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_answerable(items, retriever, top_k=5, min_score=0.0):
    rows = []

    for item in items:
        gold = snippets_for(item)
        results = retriever.search(
            item["_query"], top_k=top_k, min_score=min_score
        )

        row = {
            "query": item["_query"],
            "split": item["_split"],
            "returned": bool(results),
        }

        for k in (1, 3, 5):
            metrics = char_metrics(results, gold, k)
            row[f"p@{k}"] = metrics["precision"]
            row[f"r@{k}"] = metrics["recall"]
            row[f"f1@{k}"] = metrics["f1"]

        rows.append(row)

    return rows


def summarize(rows):
    if not rows:
        return {"n": 0}

    keys = [
        key for key in rows[0]
        if key.startswith(("p@", "r@", "f1@"))
    ]

    summary = {
        "n": len(rows),
        "abstained": sum(not row["returned"] for row in rows),
    }

    for key in keys:
        summary[key] = sum(row[key] for row in rows) / len(rows)

    return summary
