import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus
from src.retriever import Retriever

config = yaml.safe_load(
    (ROOT / "config" / "baseline.yaml").read_text(encoding="utf-8")
)["retrieval"]

tests = load_benchmark(
    ROOT / "data" / "raw" / "benchmarks" / "cuad.json"
)
docs = read_corpus(ROOT / "data" / "raw" / "corpus" / "cuad")

# Keep only documents referenced by the selected benchmark.
referenced = {
    s["file_path"]
    for item in tests
    for s in snippets_for(item)
}
docs = {k: v for k, v in docs.items() if k in referenced}

engine = Retriever(
    docs,
    config["chunk_chars"],
    config["overlap_chars"],
    config["method"],
)

item = next(x for x in tests if x["_split"] == "train")
gold = snippets_for(item)

print("QUESTION:", item["_query"])
print("\nEXPECTED ANSWERS:")
for s in gold:
    print("Document:", s["file_path"])
    print("Span:", (s["start"], s["end"]))
    print("Answer:", s["text"][:200])

print("\nTOP 5 RETRIEVED:")
results = engine.search(
    item["_query"],
    top_k=5,
    min_score=config["min_score"],
)

for i, r in enumerate(results, 1):
    print(f"\nRank {i} | Score: {r['score']:.4f}")
    print("Document:", r["doc_id"])
    print("Offsets:", (r["start"], r["end"]))
    print("Text:", r["text"][:250])

    for s in gold:
        overlap = max(
            0,
            min(r["end"], s["end"]) - max(r["start"], s["start"])
        )
        print(
            "Same document:", r["doc_id"] == s["file_path"],
            "| Character overlap:", overlap
        )