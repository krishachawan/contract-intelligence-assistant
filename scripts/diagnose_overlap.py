import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus
from src.retriever import Retriever

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

item = next(
    x for x in benchmark
    if "BicycleTX Ltd" in x.get("_query", "")
    and "governing law" in x.get("_query", "").lower()
)
gold = snippets_for(item)[0]
results = engine.search(item["_query"], top_k=5, min_score=cfg["min_score"])

print("Gold offsets:", gold["start"], gold["end"])
print("Gold clause:", docs[gold["file_path"]][gold["start"]:gold["end"]])
print("\nTop 5 retrieved chunks:")

for i, r in enumerate(results, 1):
    overlaps = (
        r["doc_id"] == gold["file_path"]
        and r["start"] < gold["end"]
        and r["end"] > gold["start"]
    )
    print(
        f"\n{i}. file={r['doc_id']}"
        f"\noffsets={r['start']}:{r['end']}"
        f"\nscore={r.get('score')}"
        f"\ngold_span_overlap={overlaps}"
        f"\ntext={r['text'][:350]!r}"
    )
