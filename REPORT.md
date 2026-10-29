# Contract Intelligence Assistant — Evaluation Report

## 1. Summary

This project implements a contract evidence retrieval system using BM25
ranking, overlapping character-based chunks, and source-document offsets.
It evaluates retrieval against the selected CUAD portion of LegalBench-RAG-mini.

The benchmark contains 194 queries: 140 training queries and 54 held-out
test queries. The selected contracts comprise 29 referenced documents.
The system returns ranked evidence passages; retrieval quality remains
a significant limitation.

## 2. System design

- **Corpus selection:** Index contracts referenced by the selected benchmark.
- **Chunking:** Split documents into character-based chunks with configurable
  overlap while preserving source offsets.
- **Retrieval:** Rank candidate passages using BM25.
- **Evidence output:** Return document identifiers, offsets, scores, and text.
- **Interface:** Provide a command-line interface and Streamlit application.
- **Evaluation:** Measure character-level precision, recall, and F1 at k=1,3,5.

## 3. Evaluation setup

- Benchmark: LegalBench-RAG-mini, CUAD subset.
- Queries selected using the project benchmark-selection procedure.
- Split: the split labels produced by the project's benchmark loader.
- Training queries: 140.
- Held-out test queries: 54.
- Referenced contracts indexed: 29.
- Gold spans checked during data validation: 272.
- Missing referenced contracts: 0.
- Span mismatches reported by validation: 0.
- Test evaluation: one recorded run for the selected configuration.

Test results were not used to select the configuration. The retrieval
results are weak and should not be interpreted as demonstrating reliable
legal-answering performance.

## 4. Retrieval results

### Training experiments

The following table reports the measured training P@5, R@5, and F1@5.
Values are proportions expressed as percentages.

| Configuration | Chunk chars | Overlap | P@5 | R@5 | F1@5 |
|---|---:|---:|---:|---:|---:|
| Baseline | 1200 | 200 | 1.35% | 13.41% | 2.38% |
| Experiment 1 | 800 | 100 | 1.25% | 6.14% | 1.97% |
| Experiment 2 | 1600 | 200 | 1.81% | 23.30% | 3.23% |
| Experiment 3 | 1200 | 300 | 1.72% | 14.79% | 2.93% |
| Experiment 4 | 800 | 200 | 0.89% | 5.53% | 1.43% |
| Experiment 5 | 1600 | 300 | 1.92% | 24.62% | 3.42% |

Experiment 5 had the strongest training P@5, R@5, and F1@5 among these
six configurations before the improvement script was introduced.
The improvement script's recorded selection should be checked in
`artifacts/improvement_history.json` after it runs.

### Held-out test results

| Metric | @1 | @3 | @5 |
|---|---:|---:|---:|
| Precision | 0.22% | 0.79% | 0.66% |
| Recall | 1.85% | 12.11% | 15.61% |
| F1 | 0.39% | 1.43% | 1.23% |

These results indicate that the current retrieval configuration frequently
fails to return the annotated evidence spans. The system is not suitable
for unsupervised legal decision-making.

## 5. Iteration history

Six chunking configurations were compared:

1. Baseline: 1200-character chunks, 200-character overlap.
2. Experiment 1: 800-character chunks, 100-character overlap.
3. Experiment 2: 1600-character chunks, 200-character overlap.
4. Experiment 3: 1200-character chunks, 300-character overlap.
5. Experiment 4: 800-character chunks, 200-character overlap.
6. Experiment 5: 1600-character chunks, 300-character overlap.

The experiments varied chunk size and overlap while keeping the retrieval
method as BM25. Experiment 5 achieved the highest training F1@5 among the
six measured configurations. This is a limited configuration search,
not evidence of broad model improvement.

The one-command loop is implemented in `scripts/improve.py`. Its saved
history and selected configuration should be verified after execution.

## 6. Error analysis

A training error-analysis run found 115 of 140 queries with incomplete
Recall@5. Several queries retrieved the correct contract but not the
specific annotated evidence passage.

For example, a governing-law query for the BicycleTX Ltd service agreement
had gold offsets 53956–54252. The correct contract appeared in the top
results, but the returned passages did not overlap the gold span.

Inspection confirmed that the source clause existed at the annotated
offsets and that chunking generated a chunk containing the clause.
This indicates a passage-ranking or evidence-selection limitation rather
than a missing source document for that example.

## 7. Abstention and unanswerable queries

The unanswerable-question file currently fails JSON parsing with an
"Extra data" error. Therefore, reliable abstention metrics have not been
established and are not reported here. The question contents must be
preserved while the file format is repaired, and the fixed set must remain
unchanged during tuning.

The current held-out metrics measure answerable-query evidence retrieval;
they do not establish that the system abstains correctly on unanswerable
questions. This remains incomplete validation.

## 8. LLM usage

The measured retrieval experiments described here use BM25 and do not
establish a measured benefit from a generative language model. No claims
about LLM accuracy, latency, or cost are made without corresponding
measurements.

## 9. Limitations and next steps

- Improve ranking of relevant clauses within the correct contract.
- Evaluate abstention using the frozen unanswerable-question set.
- Verify that all reported evaluation metrics use the intended character
  overlap definitions.
- Measure latency and resource use before making deployment claims.
- Add stronger regression tests for offsets, ranking, and abstention.
- Require human legal review before relying on retrieved contract evidence.

## 10. Reproducibility artifacts

- Configurations: `config/`
- Training metrics and per-query rows: `artifacts/`
- Held-out test metrics: `artifacts/final_test.json`
- Held-out test per-query results: `artifacts/final_test_rows.json`
- Data validation: `scripts/validate_data.py`
- Error analysis: `scripts/error_analysis.py`
- Improvement loop: `scripts/improve.py`
- Tests: `python -m pytest -q`
