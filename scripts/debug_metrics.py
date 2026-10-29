import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus
from src.retriever import Retriever
from src.evaluator import char_metrics

cfg = yaml.safe_load(
    (ROOT / "config" / "baseline.yaml").read_text(encoding="utf-8")
)["retrieval"]

tests = load_benchmark(
    ROOT / "data" / "raw" / "benchmarks" / "cuad.json"
)
docs = read_corpus(ROOT / "data" / "raw" / "corpus" / "cuad")

referenced = {
    s["file_path"]
    for item in tests
    for s in snippets_for(item)
}
docs = {k: v for k, v in docs.items() if k in referenced}

engine = Retriever(
    docs, cfg["chunk_chars"], cfg["overlap_chars"], cfg["method"]
)

# Check the same query used by the earlier diagnostic.
item = next(x for x in tests if x["_split"] == "train")
gold = snippets_for(item)
results = engine.search(
    item["_query"], top_k=cfg["top_k"], min_score=cfg["min_score"]
)

print("QUERY:", item["_query"])
print("GOLD:", gold)
print("TOP RESULT:", results[0] if results else None)
print("METRICS:", char_metrics(results, gold, 5))

if results and gold:
    print("Retrieved document:", repr(results[0]["doc_id"]))
    print("Gold document:", repr(gold[0]["file_path"]))
    print("Document IDs equal:",
          results[0]["doc_id"] == gold[0]["file_path"])