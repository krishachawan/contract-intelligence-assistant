"""Download LegalBench-RAG files from Hugging Face using huggingface_hub."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
def main():
    target = ROOT / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="awinml/legalbench-rag",
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=["benchmarks/cuad.json", "corpus/cuad/**"],
    )
    print(f"Dataset downloaded to {path}")
    print("Verify the downloaded tree; expected benchmarks/cuad.json and corpus/cuad/.")
if __name__ == "__main__":
    main()
