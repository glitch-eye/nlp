# Unified Course Assistant Evaluation

- Generated: 2026-11-19T11:19:09.436033+00:00
- Mode: real_api_required

## Parser, semantic, retrieval and deterministic RAG

# Course Assistant Evaluation

- Generated: 2026-11-19T11:19:03.405307+00:00
- Dataset: /Users/phamxuanbac/Downloads/project/nlp/sample-code-v1.0/python/nlp/app/output/samples_reviewed.jsonl
- Samples: 5
- SHA-256: `03e3a4f9a629dbc04761de93b60666f06655f05e27b4b7990d534787815a5034`
- Knowledge base: full_dataset
- KB documents: 76

## Parser

| Metric | Value |
|---|---:|
| Parse-status accuracy | 100.0% |
| Exact parse-tree match | 100.0% (5/5) |

## Semantic interpretation

| Metric | Value |
|---|---:|
| Intent accuracy | 100.0% |
| Entity precision (micro) | 100.0% |
| Entity recall (micro) | 100.0% |
| Entity F1 (micro) | 100.0% |

## Retrieval

| Metric | Value |
|---|---:|
| Hit@3 | 80.0% (4/5) |
| UNKNOWN rejection rate | n/a |
| Retrieved for insufficient-context cases | 0/0 |

## End-to-end

| Metric | Value |
|---|---:|
| Correct refusal rate | n/a |
| Valid citation rate among ANSWERED | 100.0% |
| NEEDS_REVIEW | 0 |
| Over-answered insufficient-context cases | 0 |

## Representative failures

### Parser (0 cases; showing at most 10)
- No failures recorded.

### Semantic (0 cases; showing at most 10)
- No failures recorded.

### End-to-end (1 cases; showing at most 10)
- SYN4212: chương 7 có đề cập đến PCFG không?
  - expected=EXPECTED_CONTEXT, actual status=ANSWERED, expected chunks=['topics-chapter-7-semantics-and-logical-form-6'], retrieved=['topics-chapter-3-grammar-and-parsing-2', 'topics-chapter-6-ambiguity-resolution-statistical-methods-5', 'faq-q4-3']

## Strengths observed in the implementation

- The pipeline separates parsing, semantic interpretation, retrieval and answer generation, so failures can be inspected by stage.
- The aggregate evaluation uses the offline mock by default and records dataset/runtime provenance for repeatability.
- The RAG evidence section exercises grounded answers, no-information cases and scripted Answer Guard probes.

## Limitations and error-analysis guidance

- CFG coverage is limited to patterns represented in the grammar; unseen paraphrases may be UNPARSED.
- TF-IDF and rule-based semantic signals can miss relevant chunks when wording differs substantially.
- Mock results do not measure the quality of a real LLM; citation matching also does not prove full semantic entailment.
- AUTO_LABEL_REQUIRES_REVIEW labels and expected chunk IDs must be audited before interpreting scores as ground truth.
- If the external course dataset is unavailable, trace fallback data can make retrieval metrics unrepresentative.
- Inspect parser failures as grammar/tokenization gaps, semantic failures as intent/entity mapping gaps, and retrieval/end-to-end failures as KB coverage, ranking, answer-guard or label-quality issues.

## Interpretation cautions

- Metrics measure the supplied dataset; they are not proof of production quality.
- AUTO_LABEL_REQUIRES_REVIEW labels must be manually audited before academic claims.
- Retrieval evaluation uses the original query without gold intent/entity hints.
- End-to-end evaluation uses the offline mock when launched by this CLI; it is not a real-LLM quality score.


## Generated answer evaluation

# LLM Answer Evaluation

- Generated: 2026-11-19T11:19:09.435976+00:00
- Dataset: /Users/phamxuanbac/Downloads/project/nlp/sample-code-v1.0/python/nlp/app/input/answer_eval_smoke.jsonl
- Samples: 5
- Generator: gemini-3.5-flash-lite
- Judge: disabled
- Rows with reference answers: 0
- Knowledge base: full_dataset (76 documents)

## Deterministic metrics

> Headline answerability/retrieval rates use only rows with label_status=REVIEWED and label_provenance=human_review. Other rows are diagnostic, not gold.

| Metric | Result |
|---|---:|
| Human-reviewed gold rows | 0 |
| Answer rate on answerable questions (human-reviewed only) | n/a |
| Correct refusal rate | n/a |
| Over-answered insufficient/no-context cases | 0 |
| Retrieval Hit@k | n/a |
| Retrieval micro precision / recall | n/a / n/a |
| Citation validity (checked ANSWERED only) | 100.0% |
| NEEDS_REVIEW | 0 |
| Real-model answer rows | 5 |
| Mock answer rows | 0 |
| Model fallback rows | 0 |
| Mixed real/mock rows | 0 |
| Generation calls / successful API calls | 5 / 5 |
| Mean generation latency (s) | 1.19 |
| Mean pipeline latency (s) | 1.20 |

## Model-as-judge (diagnostic only)

| Dimension | Mean score (1–5) | N |
|---|---:|---:|
| Correctness | n/a | 0 |
| Faithfulness | n/a | 0 |
| Relevancy | n/a | 0 |
| Completeness | n/a | 0 |
| Hallucination rate (judge-flagged) | n/a |
| Human-reviewed references | 0 |
| Judge evaluated / errors / skipped | 0 / 0 / 0 |

## Status confusion

| Expected → actual | Count |
|---|---:|

## Limitations
- A trace-fallback Knowledge Base is not equivalent to the full course dataset; do not report its retrieval scores as representative.
- Citation validity only checks whether a source label was among the chunks sent to the model; it does not prove semantic support.
- The answer guard's number/date checks do not detect every unsupported natural-language claim.
- LLM-as-judge is a secondary signal, not ground truth; calibrate it against human-reviewed examples.
- Correctness/completeness are scored against references only when reference_status is HUMAN_REVIEWED; unverified reference text is withheld from the judge.
- Auto-labelled and data-source-reviewed rows are not equivalent to independently human-reviewed answer gold.
- A fallback answer is not an output from the requested real model and must be excluded from real-model quality claims.

## Per-sample details

### SMOKE-001 — CO3085 có mấy tín chỉ?
- Expected: REVIEW; actual: ANSWERED
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 1.313144s; pipeline latency: 1.3157s
- Retrieved: faq-q1-0, course-info-course-id-co3085-1, course-info-course-description-2
- Guarded answer: Môn Xử lý Ngôn ngữ Tự nhiên (CO3085) có 3 tín chỉ [Nguồn: faq.txt / Q1].

### SMOKE-002 — Chương 6 nói về gì?
- Expected: REVIEW; actual: ANSWERED
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 1.157616s; pipeline latency: 1.1631s
- Retrieved: faq-q4-3, faq-q7-6, faq-q8-7
- Guarded answer: Chương 6 nói về các phương pháp thống kê [Nguồn: faq.txt / Q4].

### SMOKE-003 — Tuần 5 học những nội dung gì?
- Expected: REVIEW; actual: INSUFFICIENT_CONTEXT
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 1.033889s; pipeline latency: 1.0394s
- Retrieved: faq-q10-9, schedule-week-06-6, faq-q11-10
- Guarded answer: Các đoạn tài liệu truy xuất được không đủ thông tin để trả lời câu hỏi này, nên hệ thống không đưa ra câu trả lời để tránh suy đoán.

### SMOKE-004 — Chương 4 nói về gì và Tuần 5 học gì?
- Expected: REVIEW; actual: ANSWERED
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 1.313176s; pipeline latency: 1.32s
- Retrieved: faq-q5-4, schedule-week-06-6, faq-q7-6
- Guarded answer: - Chương 4 nói về Features và Augmented Grammar [Nguồn: faq.txt / Q5]. - Ý Tuần 5 học gì: Context không hỗ trợ cung cấp thông tin về nội dung học của Tuần 5.

### SMOKE-005 — Thời tiết TP.HCM hôm nay thế nào?
- Expected: REVIEW; actual: INSUFFICIENT_CONTEXT
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 1.134489s; pipeline latency: 1.1374s
- Retrieved: regulations-luu-y-0, faq-q15-14, faq-q14-13
- Guarded answer: Các đoạn tài liệu truy xuất được không đủ thông tin để trả lời câu hỏi này, nên hệ thống không đưa ra câu trả lời để tránh suy đoán.


## Interpretation

- Parser, semantic and retrieval metrics are evaluated separately from generated-answer quality.
- Only REVIEWED rows with label_provenance=human_review enter answerability/retrieval headline metrics.
- Smoke rows are unreviewed diagnostics and must not be presented as gold evaluation.
- LLM-as-judge is diagnostic only and is skipped for mock, fallback or mixed-model outputs.
- A real-model run is verified only when successful API call telemetry is recorded and no fallback is present on generated rows.
