
import json
import sys
import yaml
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chunker import read_corpus
from src.data_loader import load_benchmark, snippets_for
from src.retriever import Retriever
from src.evaluator import char_metrics, summarize


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "baseline.yaml"),
    )
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    config_path = Path(args.config)
    benchmark_path = ROOT / "data" / "raw" / "benchmarks" / "cuad.json"
    corpus_path = ROOT / "data" / "raw" / "corpus" / "cuad"

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = cfg["retrieval"]

    benchmark = load_benchmark(benchmark_path)
    docs = read_corpus(corpus_path)

    # Index only contracts referenced by the selected 194 questions.
    referenced = {
        s["file_path"]
        for item in benchmark
        for s in snippets_for(item)
    }
    docs = {name: text for name, text in docs.items()
            if name in referenced}

    missing = referenced - docs.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} benchmark contracts are missing. "
            f"Example: {next(iter(missing))}"
        )

    engine = Retriever(
        docs,
        chunk_chars=cfg["chunk_chars"],
        overlap_chars=cfg["overlap_chars"],
        method=cfg["method"],
    )

    train = [x for x in benchmark if x["_split"] == "train"]
    test = [x for x in benchmark if x["_split"] == "test"]

    print(f"Selected queries: {len(benchmark)}")
    print(f"Train queries: {len(train)}")
    print(f"Test queries: {len(test)}")
    print(f"Indexed contracts: {len(docs)}")
    print(f"Indexed chunks: {len(engine.chunks)}")

    rows = []
    for item in train:
        gold = snippets_for(item)
        results = engine.search(
            item["_query"],
            top_k=cfg["top_k"],
            min_score=cfg["min_score"],
        )

        row = {
            "query": item["_query"],
            "split": item["_split"],
            "returned": bool(results),
        }

        for k in (1, 3, 5):
            m = char_metrics(results, gold, k)
            row[f"p@{k}"] = m["precision"]
            row[f"r@{k}"] = m["recall"]
            row[f"f1@{k}"] = m["f1"]

        rows.append(row)

    if rows:
        print("First query direct metrics:", {
            k: rows[0][k] for k in ("p@5", "r@5", "f1@5")
        })
        print("Queries with nonzero recall@5:",
              sum(x["r@5"] > 0 for x in rows))

    report = {
        "config": cfg,
        "train": summarize(rows),
        "n_indexed_docs": len(docs),
        "n_chunks": len(engine.chunks),
        "test_queries": len(test),
        "note": "Training metrics only; test data has not been evaluated.",
    }

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)

    report["experiment"] = args.name
    report["note"] = (
        "Training metrics only; test data has not been evaluated."
    )

    (out / f"{args.name}_train.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / f"{args.name}_train_rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
