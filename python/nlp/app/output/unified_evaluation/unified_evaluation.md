# Unified Course Assistant Evaluation

- Generated: 2026-11-19T14:13:00.310403+00:00
- Mode: real_api_required

## Parser, semantic, retrieval and deterministic RAG

# Course Assistant Evaluation

- Generated: 2026-11-19T14:12:17.001066+00:00
- Dataset: /Users/phamxuanbac/Downloads/project/nlp/sample-code-v1.0/python/nlp/app/output/samples_reviewed.jsonl
- Samples: 5
- SHA-256: `6b0c4dd6a2dc4ba8fb32f26aa12cfd88260d0656f56e8f7454c2150e120d6119`
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

- Generated: 2026-11-19T14:13:00.310218+00:00
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
| Generation calls / successful API calls | 6 / 6 |
| Mean generation latency (s) | 8.64 |
| Mean pipeline latency (s) | 8.65 |

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
- Model calls: 1; generation latency: 6.032151s; pipeline latency: 6.0353s
- Retrieved: faq-q1-0, course-info-course-id-co3085-1, course-info-course-description-2
- Guarded answer: Môn Xử lý Ngôn ngữ Tự nhiên (CO3085) có 3 tín chỉ [Nguồn: faq.txt / Q1].

### SMOKE-002 — Chương 6 nói về gì?
- Expected: REVIEW; actual: ANSWERED
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 7.909637s; pipeline latency: 7.9153s
- Retrieved: faq-q4-3, schedule-week-07-7, schedule-week-08-8
- Guarded answer: Chương 6 nói về các phương pháp thống kê và bao gồm các chủ đề sau: - basic probability theory [Nguồn: schedule.txt / WEEK 07] - estimating probabilities [Nguồn: schedule.txt / WEEK 07] - part-of-speech tagging [Nguồn: schedule.txt / WEEK 07] - lexical probabilities [Nguồn: schedule.txt / WEEK 07] - probabilistic context-free grammar [Nguồn: schedule.txt / WEEK 07] - best-first parsing [Nguồn: schedule.txt / WEEK 08] - simple context-dependent best-first parser [Nguồn: schedule.txt / WEEK 08]

### SMOKE-003 — Tuần 5 học những nội dung gì?
- Expected: REVIEW; actual: ANSWERED
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 7.121596s; pipeline latency: 7.128s
- Retrieved: schedule-week-05-5, faq-q10-9, faq-q11-10
- Guarded answer: Tuần 5 học các nội dung thuộc phần Xử lý cú pháp (Syntactic Processing), chương 4, bao gồm các chủ đề: - augmented transition networks - generalized feature systems - unification grammar Và hoạt động: feature-based parsing exercises [Nguồn: schedule.txt / WEEK 05].

### SMOKE-004 — Chương 4 nói về gì và Tuần 5 học gì?
- Expected: REVIEW; actual: ANSWERED
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 2; generation latency: 14.503144s; pipeline latency: 14.5162s
- Retrieved: faq-q5-4, schedule-week-05-5, schedule-week-04-4, schedule-week-06-6, regulations-academic-integrity-2
- Guarded answer: Chương 4 nói về các chủ đề: - feature systems [Nguồn: schedule.txt / WEEK 04] - augmented grammars [Nguồn: schedule.txt / WEEK 04] - feature systems for English [Nguồn: schedule.txt / WEEK 04] - lexicon [Nguồn: schedule.txt / WEEK 04] - grammar using feature systems [Nguồn: schedule.txt / WEEK 04] - parsing with features [Nguồn: schedule.txt / WEEK 04] - augmented transition networks [Nguồn: schedule.txt / WEEK 05] - generalized feature systems [Nguồn: schedule.txt / WEEK 05] - unification grammar [Nguồn: schedule.txt / WEEK 05] Ngoài ra, chương này học về Features và Augmented Grammar [Nguồn: faq.txt / Q5]. Trong Tuần 5, nội dung học bao gồm: - Phần: Syntactic Processing - Chương: Chapter 4 - Các chủ đề: augmented transition networks, generalized feature systems, unification grammar - Hoạt động: feature-based parsing exercises [Nguồn: schedule.txt / WEEK 05]

### SMOKE-005 — Thời tiết TP.HCM hôm nay thế nào?
- Expected: REVIEW; actual: INSUFFICIENT_CONTEXT
- Model: gemini-3.5-flash-lite (real_api)
- Model calls: 1; generation latency: 7.658321s; pipeline latency: 7.6625s
- Retrieved: regulations-luu-y-0, faq-q15-14, faq-q14-13
- Guarded answer: Các đoạn tài liệu truy xuất được không đủ thông tin để trả lời câu hỏi này, nên hệ thống không đưa ra câu trả lời để tránh suy đoán.


## Interpretation

- Parser, semantic and retrieval metrics are evaluated separately from generated-answer quality.
- Only REVIEWED rows with label_provenance=human_review enter answerability/retrieval headline metrics.
- Smoke rows are unreviewed diagnostics and must not be presented as gold evaluation.
- LLM-as-judge is diagnostic only and is skipped for mock, fallback or mixed-model outputs.
- A real-model run is verified only when successful API call telemetry is recorded and no fallback is present on generated rows.
