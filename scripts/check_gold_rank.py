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

# Retrieve every chunk so we can find the rank of the gold evidence.
results = engine.search(item["_query"], top_k=len(engine.chunks), min_score=cfg["min_score"])

matches = [
    (rank, r) for rank, r in enumerate(results, 1)
    if r["doc_id"] == gold["file_path"]
    and r["start"] <= gold["start"]
    and r["end"] >= gold["end"]
]

print("Gold span:", gold["start"], gold["end"])
print("Total chunks indexed:", len(engine.chunks))
print("Gold-containing chunks found:", len(matches))

for rank, r in matches:
    print("\nGold chunk rank:", rank)
    print("Offsets:", r["start"], r["end"])
    print("Score:", r.get("score"))
    print("Text:", repr(r["text"][:700]))

print("\nTop 10 results:")
for rank, r in enumerate(results[:10], 1):
    print(rank, r["doc_id"], r["start"], r["end"], repr(r["text"][:120]))
