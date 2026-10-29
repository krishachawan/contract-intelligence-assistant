
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.chunker import read_corpus
from src.data_loader import load_benchmark, snippets_for
from src.retriever import Retriever
from src.evaluator import char_metrics

def main():
    cfg = yaml.safe_load(
        (ROOT / "config" / "best.yaml").read_text(encoding="utf-8")
    )["retrieval"]

    benchmark = load_benchmark(
        ROOT / "data" / "raw" / "benchmarks" / "cuad.json"
    )
    docs = read_corpus(ROOT / "data" / "raw" / "corpus" / "cuad")

    referenced = {
        s["file_path"]
        for item in benchmark
        for s in snippets_for(item)
    }
    docs = {k: v for k, v in docs.items() if k in referenced}
    engine = Retriever(
        docs,
        chunk_chars=cfg["chunk_chars"],
        overlap_chars=cfg["overlap_chars"],
        method=cfg["method"],
    )

    failures = []
    for item in benchmark:
        if item["_split"] != "train":
            continue

        gold = snippets_for(item)
        results = engine.search(
            item["_query"],
            top_k=5,
            min_score=cfg["min_score"],
        )
        metrics = char_metrics(results, gold, 5)

        if metrics["recall"] < 1.0:
            failures.append({
                "query": item["_query"],
                "recall_at_5": metrics["recall"],
                "precision_at_5": metrics["precision"],
                "f1_at_5": metrics["f1"],
                "gold_documents": [g["file_path"] for g in gold],
                "retrieved_documents": [r["doc_id"] for r in results],
                "retrieved_passages": [
                    {
                        "doc_id": r["doc_id"],
                        "start": r["start"],
                        "end": r["end"],
                        "text": r["text"][:500],
                    }
                    for r in results
                ],
            })

    failures.sort(key=lambda x: x["recall_at_5"])
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    path = out / "error_analysis_top10.json"
    path.write_text(
        json.dumps(failures[:10], indent=2),
        encoding="utf-8",
    )

    print(f"Training queries with incomplete recall@5: {len(failures)}")
    print(f"Saved up to 10 examples to: {path}")
    for i, row in enumerate(failures[:10], 1):
        print(
            f"{i}. Recall@5={row['recall_at_5']:.3f} | "
            f"Query: {row['query']}"
        )

if __name__ == "__main__":
    main()
