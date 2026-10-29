import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus
from src.retriever import Retriever
from src.evaluator import char_metrics

cfg = yaml.safe_load(
    (ROOT / "config/best.yaml").read_text(encoding="utf-8")
)["retrieval"]

benchmark = load_benchmark(ROOT / "data/raw/benchmarks/cuad.json")
docs = read_corpus(ROOT / "data/raw/corpus/cuad")
referenced = {s["file_path"] for item in benchmark for s in snippets_for(item)}
docs = {k: v for k, v in docs.items() if k in referenced}

engine = Retriever(
    docs,
    chunk_chars=cfg["chunk_chars"],
    overlap_chars=cfg["overlap_chars"],
    method=cfg["method"],
)

test = [x for x in benchmark if x["_split"] == "test"]
rows = []

for item in test:
    gold = snippets_for(item)
    row = {"query": item["_query"]}
    for k in (1, 3, 5):
        results = engine.search(
            item["_query"], top_k=k, min_score=cfg["min_score"]
        )
        m = char_metrics(results, gold, k)
        for metric in ("precision", "recall", "f1"):
            row[f"{metric}@{k}"] = m[metric]
    rows.append(row)

metrics = {}
for k in (1, 3, 5):
    for metric in ("precision", "recall", "f1"):
        key = f"{metric}@{k}"
        metrics[key] = sum(r[key] for r in rows) / len(rows) if rows else 0

out = ROOT / "artifacts"
out.mkdir(exist_ok=True)
(out / "final_test_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
(out / "final_test.json").write_text(json.dumps({
    "experiment": "best",
    "split": "test",
    "test_queries": len(test),
    "metrics": metrics,
    "config": cfg
}, indent=2), encoding="utf-8")

print("HELD-OUT TEST RESULTS")
print("Test queries:", len(test))
for k in (1, 3, 5):
    print(f"@{k}: P={metrics[f'precision@{k}']:.4f} "
          f"R={metrics[f'recall@{k}']:.4f} F1={metrics[f'f1@{k}']:.4f}")
print("Saved artifacts/final_test.json and final_test_rows.json")
