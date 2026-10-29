"""Train-only deterministic parameter search. Logs every candidate; no test data is read."""
import json, yaml
from pathlib import Path
from src.chunker import read_corpus
from src.data_loader import load_benchmark
from src.retriever import Retriever
from src.evaluator import evaluate_answerable, summarize
ROOT = Path(__file__).resolve().parent

def main():
    bench = load_benchmark(ROOT / "data/raw/benchmarks/cuad.json")
    train = [x for x in bench if x["_split"] == "train"]
    docs = read_corpus(ROOT / "data/raw/corpus")
    referenced = {s.get("file_path") for item in bench for s in item.get("snippets", [])}
    docs = {k:v for k,v in docs.items() if k in referenced}
    if not docs: raise SystemExit("No referenced CUAD documents found. Check dataset paths.")
    candidates = [
      {"chunk_chars": 800, "overlap_chars": 100, "top_k": 1, "min_score": 0.0},
      {"chunk_chars": 800, "overlap_chars": 100, "top_k": 3, "min_score": 0.0},
      {"chunk_chars": 1200, "overlap_chars": 200, "top_k": 3, "min_score": 0.0},
      {"chunk_chars": 1200, "overlap_chars": 200, "top_k": 5, "min_score": 0.0},
      {"chunk_chars": 1800, "overlap_chars": 250, "top_k": 5, "min_score": 0.0},
      {"chunk_chars": 1800, "overlap_chars": 250, "top_k": 5, "min_score": 1.0},
    ]
    log_path = ROOT / "experiments/experiment_log.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    best = None
    with log_path.open("w", encoding="utf-8") as log:
      for i, c in enumerate(candidates, 1):
        engine = Retriever(docs, c["chunk_chars"], c["overlap_chars"])
        rows = evaluate_answerable(train, engine, c["top_k"], c["min_score"])
        metrics = summarize(rows)
        # Primary metric: mean character recall@5; guardrail: precision@5 must not fall > 0.05.
        objective = metrics.get("r@5", 0.0)
        accepted = best is None or (objective > best["objective"] and metrics.get("p@5",0) >= best["p@5"] - 0.05)
        record = {"iteration":i,"proposer":"deterministic parameter sweep","change":c,"reason":"test chunk size/overlap/top-k/score threshold using train only","metrics":metrics,"objective":objective,"accepted":accepted,"split":"train only"}
        log.write(json.dumps(record)+"\n")
        if accepted: best = {"objective":objective,"p@5":metrics.get("p@5",0),"config":c,"metrics":metrics}
        print(json.dumps(record))
    if best:
      final_cfg={"retrieval":{"method":"bm25",**best["config"],"tokenizer":"simple"}}
      (ROOT/"config/final.yaml").write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding="utf-8")
      print("Best train-only configuration written to config/final.yaml")
if __name__ == "__main__": main()
