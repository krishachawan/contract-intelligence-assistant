def classify(results: list[dict], min_score: float) -> list[dict]:
    """Filter results below a threshold; threshold must be calibrated on train only."""
    return [r for r in results if r.get("score", float("-inf")) >= min_score]
