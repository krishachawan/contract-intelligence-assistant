from src.evaluator import char_metrics


def test_overlapping_chunk_gets_nonzero_credit():
    results = [
        {
            "doc_id": "cuad/example.txt",
            "start": 0,
            "end": 990,
        }
    ]

    gold = [
        {
            "file_path": "cuad/example.txt",
            "start": 598,
            "end": 989,
            "text": "example answer",
        }
    ]

    metrics = char_metrics(results, gold, k=1)

    assert metrics["precision"] > 0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] > 0