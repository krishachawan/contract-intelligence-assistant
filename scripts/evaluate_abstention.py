import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus
from src.retriever import Retriever

def load_questions(path):
    raw = path.read_text(encoding="utf-8").strip()
    # Handle the literal trailing characters \n if present.
    if raw.endswith(r"\n"):
        raw = raw[:-2].rstrip()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    return data

def main():
    question_path = ROOT / "data" / "unanswerable_queries.json"
    questions = load_questions(question_path)

    cfg = yaml.safe_load(
        (ROOT / "config" / "final.yaml").read_text(encoding="utf-8")
    )["retrieval"]

    benchmark = load_benchmark(ROOT / "data/raw/benchmarks/cuad.json")
    docs = read_corpus(ROOT / "data/raw/corpus/cuad")
    referenced = {
        s["file_path"] for item in benchmark for s in snippets_for(item)
    }
    docs = {k: v for k, v in docs.items() if k in referenced}

    engine = Retriever(
        docs,
        chunk_chars=cfg["chunk_chars"],
        overlap_chars=cfg["overlap_chars"],
        method=cfg["method"],
    )

    rows = []
    for q in questions:
        query = q.get("query", "")
        results = engine.search(
            query,
            top_k=cfg.get("top_k", 5),
            min_score=cfg.get("min_score", 0.0),
        )
        rows.append({
            "id": q.get("id"),
            "query": query,
            "split": q.get("split", "unknown"),
            "expected_answerable": q.get("answerable"),
            "abstained": len(results) == 0,
            "returned_passages": len(results),
            "top_score": results[0]["score"] if results else None,
            "top_source": results[0]["doc_id"] if results else None,
        })

    total = len(rows)
    abstained = sum(r["abstained"] for r in rows)
    summary = {
        "question_count": total,
        "abstained": abstained,
        "did_not_abstain": total - abstained,
        "abstention_rate": abstained / total if total else 0,
        "config": cfg,
        "by_split": {},
        "warning": (
            "Abstention is defined as returning zero passages under the "
            "configured score threshold. This does not verify semantic "
            "correctness of the retrieved passages."
        ),
    }

    for split in sorted(set(r["split"] for r in rows)):
        subset = [r for r in rows if r["split"] == split]
        n = len(subset)
        count = sum(r["abstained"] for r in subset)
        summary["by_split"][split] = {
            "count": n,
            "abstained": count,
            "did_not_abstain": n - count,
            "abstention_rate": count / n if n else 0,
        }

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "unanswerable_eval_rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (out / "unanswerable_eval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("UNANSWERABLE QUESTION EVALUATION")
    print("Questions:", total)
    print(f"Abstained: {abstained}/{total} ({summary['abstention_rate']:.1%})")
    print(f"Returned passages: {total-abstained}/{total}")
    for split, values in summary["by_split"].items():
        print(
            f"{split}: {values['abstained']}/{values['count']} abstained "
            f"({values['abstention_rate']:.1%})"
        )
    print("Saved artifacts/unanswerable_eval.json")
    print("Saved artifacts/unanswerable_eval_rows.json")

if __name__ == "__main__":
    main()
