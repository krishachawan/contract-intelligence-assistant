import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_benchmark, snippets_for
from src.chunker import read_corpus

benchmark = load_benchmark(ROOT / "data/raw/benchmarks/cuad.json")
docs = read_corpus(ROOT / "data/raw/corpus/cuad")

target = next(
    item for item in benchmark
    if "BicycleTX Ltd" in item.get("_query", "")
    and "governing law" in item.get("_query", "").lower()
)

print("QUERY:", target["_query"])

for gold in snippets_for(target):
    path = gold["file_path"]
    start, end = gold["start"], gold["end"]
    print("\nGOLD FILE:", path)
    print("GOLD OFFSETS:", start, end)
    print("GOLD TEXT:", repr(docs[path][start:end][:1500]))
