import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("baseline", "config/baseline.yaml"),
    ("experiment1", "config/experiment1.yaml"),
    ("experiment2", "config/experiment2.yaml"),
    ("experiment3", "config/experiment3.yaml"),
    ("experiment4", "config/experiment4.yaml"),
    ("experiment5", "config/experiment5.yaml"),
]

def mean_f1_at_5(rows):
    if not rows:
        raise ValueError("No training evaluation rows found.")
    values = []
    for row in rows:
        if "f1@5" in row:
            values.append(float(row["f1@5"]))
        elif "f1_5" in row:
            values.append(float(row["f1_5"]))
        else:
            raise KeyError(
                "Cannot find F1@5 in row keys: " + str(list(row.keys()))
            )
    return sum(values) / len(values)

def main():
    history = []
    for name, config in CONFIGS:
        print(f"\n=== Evaluating {name} with {config} ===")
        subprocess.run(
            [sys.executable, "evaluate.py", "--config", config, "--name", name],
            cwd=ROOT,
            check=True,
        )
        rows_path = ROOT / "artifacts" / f"{name}_train_rows.json"
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        score = mean_f1_at_5(rows)
        history.append({
            "iteration": name,
            "config": config,
            "training_queries": len(rows),
            "mean_f1_at_5": score,
        })
        print(f"Training F1@5: {score:.6f}")

    best = max(history, key=lambda x: x["mean_f1_at_5"])
    shutil.copy2(ROOT / best["config"], ROOT / "config" / "best.yaml")

    out = ROOT / "artifacts"
    (out / "improvement_history.json").write_text(
        json.dumps({"selection_metric": "training mean F1@5",
                    "iterations": history, "selected": best}, indent=2),
        encoding="utf-8",
    )
    print("\nSelected:", best["iteration"])
    print("Best config copied to config/best.yaml")
    print("History saved to artifacts/improvement_history.json")

if __name__ == "__main__":
    main()
