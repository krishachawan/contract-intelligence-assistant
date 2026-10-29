import json
from pathlib import Path
from statistics import median

p = Path("artifacts/unanswerable_eval_rows.json")
rows = json.loads(p.read_text(encoding="utf-8"))
train = [r for r in rows if r.get("split") == "train" and r.get("top_score") is not None]

print("Unanswerable training scores:", len(train))
print("Minimum:", round(min(r["top_score"] for r in train), 3))
print("Median:", round(median(r["top_score"] for r in train), 3))
print("Maximum:", round(max(r["top_score"] for r in train), 3))
print()
print("IMPORTANT: top_score is the adjusted score, not the original BM25 score.")
print("Do not use these values directly as the min_score threshold.")
print("Next, inspect the evaluation code to calculate comparable original scores.")
