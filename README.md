# Contract Intelligence Assistant

A reproducible legal-contract passage retrieval prototype for the CUAD subset of LegalBench-RAG. It returns ranked passages with source paths and character offsets and can abstain when no result passes a configured score threshold. It is a retrieval aid, not legal advice.

## Setup

Requires Python 3.10+ (Python 3.11 recommended).

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/create_unanswerable.py
```

Inspect `data/raw/` after download. The downloader requests only `benchmarks/cuad.json` and `corpus/cuad/**`. The corpus loader currently reads `.txt` and `.md`; if the dataset mirror uses another text format, adapt `src/chunker.py` and test offsets before evaluation.

## Run

```bash
python evaluate.py
python improve.py
python cli.py "When does the exclusivity period end?"
streamlit run app.py
pytest -q
```

## Architecture

- `src/data_loader.py`: prescribed mini selection and deterministic split.
- `src/chunker.py`: chunks documents while preserving source character offsets.
- `src/retriever.py`: BM25 retrieval.
- `src/evaluator.py`: character-level span precision/recall/F1.
- `config/*.yaml`: pipeline parameters.
- `app.py` and `cli.py`: interfaces.
- `scripts/create_unanswerable.py`: initial fixed negative-query set.
- `evaluate.py`: baseline training evaluation.

## Reproducing results

Run the commands above from the repository root. `evaluate.py` writes `artifacts/baseline_train.json`. The current scaffold is a starting implementation; do not report metrics until the benchmark and corpus paths, annotation schema, character offsets, and evaluator have been validated against the downloaded data. The final report must include exact sample sizes, test-run count, actual results, and failures.

## Assumptions and limitations

- CUAD's selected mini subset is produced with the selection function in the assignment.
- Split assignment hashes the stripped query exactly as specified.
- Only documents referenced by selected queries should be indexed. Verify path matching after download.
- BM25 is the initial baseline; it does not resolve definitions or cross-references and is not guaranteed to abstain correctly without threshold calibration on training data.
- Character-level metrics should be validated against the benchmark authors' definition before final reporting.
- The 36 generated negative queries are candidates, not a substitute for verifying that the corpus truly does not answer them. Commit the reviewed file before improvement iterations and do not edit it afterward.
- Do not use test results to choose parameters.

## AI tool usage

I have used ChatGPT as an AI-assisted coding tool during the development of this project. However, I independently understood the overall system architecture, including the different components, their responsibilities, and how they interact with each other. I also reviewed the generated code, made necessary modifications, tested the implementation, and ensured that I understood the functionality of the code used in the project. I can explain the design decisions, implementation, and technical aspects of the system.
