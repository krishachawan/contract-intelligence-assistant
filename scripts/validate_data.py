import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus

benchmark_path = ROOT / "data" / "raw" / "benchmarks" / "cuad.json"
corpus_path = ROOT / "data" / "raw" / "corpus" / "cuad"

tests = load_benchmark(benchmark_path)
docs = read_corpus(corpus_path)

referenced = {
    s["file_path"]
    for item in tests
    for s in snippets_for(item)
}

print("Selected benchmark queries:", len(tests))
print("Unique referenced contracts:", len(referenced))
print("Loaded contract files:", len(docs))
print("Missing referenced contracts:", len(referenced - docs.keys()))

checked = 0
mismatches = []

for item in tests:
    for snippet in snippets_for(item):
        path = snippet["file_path"]
        if path not in docs:
            continue

        text = docs[path]
        start, end = snippet["start"], snippet["end"]
        actual = text[start:end]

        checked += 1
        if start < 0 or end < start or actual != snippet["text"]:
            mismatches.append({
                "query": item["_query"],
                "file": path,
                "start": start,
                "end": end,
                "expected": snippet["text"][:100],
                "actual": actual[:100],
            })

print("Spans checked:", checked)
print("Span mismatches:", len(mismatches))

for mismatch in mismatches[:5]:
    print("\nMISMATCH:", mismatch)