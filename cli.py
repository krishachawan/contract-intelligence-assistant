import argparse, yaml
from pathlib import Path
from src.chunker import read_corpus
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parent
def main():
    p = argparse.ArgumentParser(description="Contract Intelligence Assistant")
    p.add_argument("query", help="Natural-language contract question")
    p.add_argument("--config", default="config/final.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())["retrieval"]
    docs = read_corpus(ROOT / "data" / "raw" / "corpus")
    if not docs: raise SystemExit("Dataset missing. Run: python scripts/download_data.py")
    engine = Retriever(docs, cfg["chunk_chars"], cfg["overlap_chars"], cfg["method"])
    results = engine.search(args.query, cfg["top_k"], cfg["min_score"])
    if not results:
        print("No sufficiently relevant passage found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\\n[{i}] score={r['score']:.4f}\\nSource: {r['doc_id']}\\nOffsets: [{r['start']}:{r['end']}]\\n{r['text']}")
if __name__ == "__main__":
    main()
