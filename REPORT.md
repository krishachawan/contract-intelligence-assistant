# Contract Intelligence Assistant --- Evaluation Report

## 1. Project Summary

The Contract Intelligence Assistant is a retrieval-based tool for
answering natural-language questions about a limited collection of
contracts. It retrieves ranked text passages from the indexed corpus and
reports the source document and character offsets so that users can
inspect the supporting evidence.

The implementation uses BM25 retrieval, configurable character-based
chunking, a command-line interface, a Streamlit interface, evaluation
scripts, and automated tests.

## 2. Scope and Dataset

The benchmark loader selected **194 queries** and identified **29 unique
contracts** referenced by those queries. Dataset validation reported:

-   Selected benchmark queries: 194
-   Unique referenced contracts: 29
-   Contract files available in the downloaded corpus: 462
-   Missing referenced contracts: 0
-   Gold answer spans checked: 272
-   Gold span mismatches: 0

The benchmark evaluation used 140 training queries and 54 held-out test
queries. An additional set of 36 manually authored unanswerable
questions was used to examine abstention behavior (28 training questions
and 8 test questions).

The data validation check confirmed that all referenced contracts were
found and that the 272 checked gold spans matched the source text. The
reported retrieval experiments used the selected benchmark contracts.

## 3. System Design

### 3.1 Document loading and chunking

Contract text is loaded from the corpus and split into character-based
chunks. Each chunk retains its document identifier and start/end
character offsets. These offsets allow retrieved passages to be traced
back to their location in the source contract.

The tested configurations varied chunk length and overlap. The baseline
used 1,200-character chunks with 200-character overlap. The best
training configuration among the tested chunking experiments used
1,600-character chunks with 300-character overlap.

### 3.2 Retrieval

The retriever uses BM25 to rank chunks against a natural-language query.
It returns the document identifier, offsets, passage text, and retrieval
score. A query-specific amendment-number boost is also implemented when
the query specifies an amendment and the chunk contains an exact
amendment-number match plus sufficient contextual term overlap.

### 3.3 Interfaces and evaluation

The project provides: - `cli.py` for command-line queries - `app.py` for
the Streamlit interface - `evaluate.py` and supporting scripts for
benchmark evaluation - YAML configuration files for baseline and
experimental retrieval settings - Automated tests for core retrieval and
evaluation behavior

## 4. Experimental Setup

The baseline configuration used 1,200-character chunks, 200-character
overlap, and top-5 retrieval. Five chunking configurations were
compared. The reported metrics are passage-level character precision,
recall, and F1 at rank 5, as emitted by the project evaluator.

The experiment with 1,600-character chunks and 300-character overlap
achieved the strongest training F1@5 among the tested configurations.
The held-out test evaluation was subsequently run using
`config/best.yaml`.

## 5. Results

### 5.1 Training experiments

  ---------------------------------------------------------------------------------
  Configuration   Chunk length      Overlap   Precision@5     Recall@5         F1@5
  --------------- ------------ ------------ ------------- ------------ ------------
  Baseline               1,200          200         1.35%       13.41%        2.38%

  Experiment 1             800          100         1.25%        6.14%        1.97%

  Experiment 2           1,600          200         1.81%       23.30%        3.23%

  Experiment 3           1,200          300         1.72%       14.79%        2.93%

  Experiment 4             800          200         0.89%        5.53%        1.43%

  Experiment 5           1,600          300         1.92%       24.62%        3.42%
  ---------------------------------------------------------------------------------

Experiment 5 produced the highest training F1@5 in this comparison.
These values are training results and should not be interpreted as
held-out performance.

### 5.2 Held-out test results

Using `config/best.yaml`, evaluation on the 54 held-out benchmark
queries reported:

  Metric     Precision   Recall      F1
  -------- ----------- -------- -------
  @1             0.22%    1.85%   0.39%
  @3             0.79%   12.11%   1.43%
  @5             0.66%   15.61%   1.23%

These results indicate that retrieval quality remains limited. The top-5
recall is higher than at rank 1, but precision and F1 are low. The
training-to-test difference also shows that the best training
configuration does not translate into strong held-out results.

### 5.3 Unanswerable-query evaluation

The initial abstention evaluation covered 36 manually authored
unanswerable questions. With the evaluated configuration, the system
abstained on **0 of 36 questions (0.0%)** and returned passages for all
36 questions.

  Split        Questions   Abstained   Abstention rate
  ---------- ----------- ----------- -----------------
  Training            28           0              0.0%
  Test                 8           0              0.0%
  Total               36           0              0.0%

This is a failed abstention result, not evidence of successful
rejection. The current BM25 score alone has not been shown to reliably
distinguish answerable from unanswerable queries. A score threshold must
be calibrated using training data and validated on held-out questions
before the system can claim effective abstention. BM25 scores can also
be negative, so a threshold of zero should not be treated as a universal
relevance boundary.

## 6. Data Integrity and Validation

The dataset validation script reported no missing referenced contracts
and no mismatches across the 272 checked gold spans. The project also
includes data-processing and diagnostic scripts for inspecting chunking
and retrieval behavior.

The data validation results support the integrity of the checked source
offsets. They do not, by themselves, establish that retrieval is
accurate or that all possible leakage and near-duplicate risks have been
eliminated. Those concerns should remain explicit in future evaluation.

## 7. Automated Tests and Execution

The automated test suite was run after correcting the score-filter
behavior. The latest recorded result was:

**4 passed**

The command-line help check also succeeded. A sample CLI query returned
ranked passages with source paths and character offsets. However, the
sample query returned passages from different contracts, illustrating
that successful execution does not guarantee answer relevance.

Typical commands:

``` powershell
python -m pip install -r requirements.txt
python -m pytest -q
python cli.py "What is the expiration date of the contract?" --config config/final.yaml
streamlit run app.py
```

The exact configuration and data paths should be checked against the
README before running in a fresh environment.

## 8. Limitations

1.  **Low retrieval precision and F1:** the measured held-out results
    indicate that many returned character ranges do not overlap the gold
    answer spans.
2.  **Abstention is not effective:** the system returned passages for
    all 36 unanswerable questions in the measured evaluation.
3.  **Training/test performance differs:** the best training
    configuration performed worse on held-out queries.
4.  **BM25 is lexical:** lexical overlap can rank passages that share
    terms with a query but do not answer it.
5.  **Limited evaluation sample:** the held-out benchmark contains 54
    queries, and the unanswerable test split contains 8 questions;
    results on these samples may not generalize to broader contract
    collections.
6.  **Further validation needed:** successful tests and valid source
    offsets do not establish semantic answer correctness, robust
    abstention, or production readiness.

## 9. Recommended Next Steps

-   Inspect the highest-ranked false positives and improve
    query-to-passage relevance.
-   Calibrate abstention using training-only answerable and unanswerable
    examples, keeping the test split untouched until final evaluation.
-   Consider a separate relevance classifier or semantic reranker rather
    than relying on a raw BM25 score threshold alone.
-   Add tests for answerable and unanswerable queries, including
    negative BM25 scores and queries with no corpus support.
-   Re-run the held-out evaluation after any change and report the new
    metrics without replacing the earlier results.
-   Verify the exact data split procedure and document
    duplicate/near-duplicate checks before claiming leakage-free
    evaluation.

## 10. Conclusion

The project provides a functioning BM25-based contract passage retrieval
prototype with source identifiers, character offsets, a CLI, a Streamlit
interface, configurable experiments, and passing core tests. Data
validation found all 29 referenced contracts and verified 272 gold
spans.

The measured retrieval results are nevertheless weak, and the current
abstention evaluation rejected none of the 36 unanswerable questions.
The system should therefore be presented as a working prototype with
documented evaluation limitations---not as a reliably accurate
contract-answering or abstention system. Improving relevance ranking and
validating abstention on held-out data are the main priorities.
