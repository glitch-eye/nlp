# Core Evaluation Module

This branch adds an evaluation layer for the current **single-intent mainline**.
It is intentionally independent of the experimental multi-intent work.

## Scope

The evaluator measures four stages:

1. **Parser / CFG**
   - Parse-status accuracy against `parse_status` labels.
   - Exact parse-tree match against the stored `parse_tree` baseline.
   - Per-sample failures for grammar regression inspection.
2. **Semantic interpretation**
   - Intent accuracy.
   - Entity precision, recall and micro-F1.
   - Missing and unexpected entity IDs per sample.
3. **Retrieval**
   - Existing retrieval evaluator, using the original query only.
   - Hit@k against `expected_chunk_ids`.
   - UNKNOWN rejection and retrieval behaviour on insufficient-context cases.
   - Gold intent/entity labels are not injected into the query path.
4. **End-to-end**
   - Answerability/status confusion.
   - Retrieval hit rate on answerable questions.
   - Correct refusal rate, valid citation rate and over-answering.
   - The CLI uses the offline mock provider; these results are not a real-LLM quality score.

## Dataset

Default input: `nlp/app/output/samples_labeled.jsonl`.

Expected fields:
- `query` (required string)
- `sample_id`
- `parse_status` and optionally `parse_tree`
- `intent`
- `entities` (list of entity IDs)
- `expected_chunk_ids` (list of chunk IDs)
- `expected_answerability`: `EXPECTED_CONTEXT`, `INSUFFICIENT_CONTEXT`, `NO_RELEVANT_CONTEXT`, or `REVIEW`
- `label_status`

The current generated dataset contains `AUTO_LABEL_REQUIRES_REVIEW` rows. Do not treat aggregate scores as ground truth until labels and expected chunk IDs have been manually audited. The report includes a SHA-256 digest and label counts to make the evaluated dataset traceable.

## Run

From the `python/` directory:

```bash
python -m nlp.app.evaluation
```

Optional:

```bash
python -m nlp.app.evaluation \
  --samples nlp/app/output/samples_labeled.jsonl \
  --data-dir /path/to/NLP_Course_Assistant_Sample_Data \
  --top-k 3 --min-score 0.08
```

Skip selected stages when debugging:

```bash
python -m nlp.app.evaluation --skip-e2e
python -m nlp.app.evaluation --skip-retrieval --skip-e2e
```

Use `--max-samples N` for a quick smoke run. It takes an evenly spaced subset; omit it for the full dataset.

## Outputs

By default, reports are written to `nlp/app/output/evaluation/`:

- `evaluation_report.json`: full metrics, per-case failures, confusion tables, runtime and dataset provenance.
- `evaluation_report.md`: compact human-readable summary.

Generated reports are run artifacts and should not be committed unless the project explicitly wants to version a particular evaluation snapshot.

## Metric interpretation

- **Parse accuracy** checks whether the parser accepts/rejects a sample as labelled.
- **Exact tree match** detects structural grammar changes even when a sentence remains parseable.
- **Intent accuracy** is exact label match.
- **Entity precision/recall/F1** use micro aggregation across entity IDs.
- **Hit@k** is true if at least one expected chunk appears in the top-k retrieved chunks.
- **Refusal accuracy** measures refusal on rows whose expected answerability is not `EXPECTED_CONTEXT`; inspect `REVIEW` rows separately before drawing conclusions.
- **Citation validity** is measured only among answers whose status is `ANSWERED`.

This is a diagnostic framework, not a substitute for manually reviewed test labels, error analysis, or a real-user evaluation.
