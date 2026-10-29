from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any

def mini(tests: list[dict], n: int = 194) -> list[dict]:
    """Exact deterministic selection rule supplied by the assignment."""
    if len(tests) <= n:
        return tests
    def key(t: dict) -> float:
        random.seed(t["snippets"][0]["file_path"])
        return random.random()
    return sorted(tests, key=key)[:n]

def assign_split(query: str) -> str:
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"

def load_benchmark(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tests = mini(raw["tests"], 194)
    rows = []
    for item in tests:
        query = item.get("query") or item.get("question") or item.get("text")
        if not query:
            raise ValueError(f"Could not find query text in benchmark item keys: {list(item)}")
        rows.append({**item, "_query": query, "_split": assign_split(query)})
    return rows

def snippets_for(item: dict) -> list[dict]:
    """Read the benchmark's annotated character spans."""
    out = []

    for s in item.get("snippets", []):
        span = s.get("span", [-1, -1])

        out.append({
            "file_path": s.get("file_path", "").replace("\\", "/"),
            "start": int(span[0]),
            "end": int(span[1]),
            "text": s.get("answer", ""),
        })

    return out
