import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus, chunk_text

benchmark = load_benchmark(ROOT / "data/raw/benchmarks/cuad.json")
docs = read_corpus(ROOT / "data/raw/corpus/cuad")

item = next(
    x for x in benchmark
    if "BicycleTX Ltd" in x.get("_query", "")
    and "governing law" in x.get("_query", "").lower()
)

gold = snippets_for(item)[0]
text = docs[gold["file_path"]]
clause = "This Agreement shall be governed by and interpreted"

print("Gold offsets:", gold["start"], gold["end"])
print("Clause position in original:", text.find(clause))
print("Original clause:", repr(text[gold["start"]:gold["end"]]))

chunks = chunk_text(
    gold["file_path"], text,
    chunk_chars=500, overlap_chars=100
)

covering = [
    c for c in chunks
    if c.start < gold["end"] and c.end > gold["start"]
]

print("\nTotal chunks:", len(chunks))
print("Chunks overlapping gold span:", len(covering))

for c in covering:
    print("\nChunk offsets:", c.start, c.end)
    print("Contains clause wording:", clause in c.text)
    print("Gold span fully contained:", c.start <= gold["start"] and c.end >= gold["end"])
    print(repr(c.text[:700]))
