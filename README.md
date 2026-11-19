# Course Assistant NLP — CO3085

Trợ lý hỗ trợ học tập cho môn **Xử lý Ngôn ngữ Tự nhiên (CO3085)**. Repository triển khai pipeline từ CFG/Earley Parser và Semantic Frame đến hybrid retrieval, RAG, Answer Guard, hội thoại nhiều lượt và đánh giá.

> Tài liệu cài đặt, kiến trúc và hướng dẫn sử dụng đầy đủ: [python/README.md](python/README.md).  
> Tài liệu kỹ thuật theo module: [python/nlp/app/README.md](python/nlp/app/README.md).  
> Thiết kế multi-intent: [python/GRAMMAR_V5_MULTI_INTENT.md](python/GRAMMAR_V5_MULTI_INTENT.md).

## Repository layout

~~~text
nlp/
├── README.md
└── python/
    ├── README.md
    ├── GRAMMAR_V5_MULTI_INTENT.md
    ├── requirements.txt
    ├── setup.py
    ├── data/                 # Course knowledge base
    ├── nlp/app/              # Application, CLI, API and evaluators
    │   ├── input/            # Evaluation/dialogue inputs
    │   └── output/           # Generated local reports and traces
    └── test/                 # unittest suite
~~~

## Requirements

- Python 3.10+ recommended (the package declares Python 3.8+ compatibility).
- macOS, Linux or Windows.
- Network/API key only when using a real OpenAI-compatible model.

## Install

Run from the repository root:

~~~bash
cd python
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

The course data directory is expected at python/data/. If using an external course-data checkout, pass its path with --data-dir.

## Quick start

All commands below run from python/.

~~~bash
# Run the assistant with the offline mock provider
python -m nlp.app.run_assistant --question "Chương 6 nói về gì?"

# Interactive multi-turn dialogue
python -m nlp.app.run_assistant --interactive

# Start the HTTP API
python -m nlp.app.api --host 127.0.0.1 --port 8000

# Run the test suite
python -m unittest discover -s test -v
~~~

## Use a real LLM

The default provider is the deterministic offline mock. To use an OpenAI-compatible Chat Completions endpoint:

~~~bash
export OPENAI_API_KEY="YOUR_API_KEY"
export LLM_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m nlp.app.run_assistant --provider openai
~~~

Never commit API keys. An API failure may trigger a clearly marked mock fallback; therefore selecting --provider openai alone does not prove that an answer came from the real model. Use the evaluation telemetry and --require-real-model when a run must verify successful real API generation.

## Evaluation

~~~bash
# Parser / semantic / retrieval / deterministic end-to-end evaluation
python -m nlp.app.evaluation

# Real answer-generation evaluation (mock by default)
python -m nlp.app.answer_evaluation --provider mock

# Unified report; requires OPENAI_API_KEY for a verified real-model run
python -m nlp.app.unified_evaluation --provider openai --require-real-model --max-samples 5
~~~

The unified evaluator writes JSON and Markdown reports under nlp/app/output/unified_evaluation/. The real-model requirement fails when a row did not make a successful generation API call (for example, when the pre-generation Answer Guard correctly stopped it) or when any generation call fell back to mock.

For evaluation datasets, distinguish human-reviewed labels from automatically generated labels. Synthetic/auto-labelled samples are useful for regression and coverage checks but must not be reported as an independent human-reviewed gold set. See the full README for dataset schemas, metrics, stratified sampling, judge settings and limitations.

## Multi-intent behavior

Grammar v5 detects compound questions only when it can partition the input into independently CFG-valid clauses. Each clause then runs through the existing semantic, retrieval and answer-generation path; the resulting answers, evidence, sources and entities are merged. This is **per-clause processing**, not a single aggregated LLM request for the entire multi-intent question. See [GRAMMAR_V5_MULTI_INTENT.md](python/GRAMMAR_V5_MULTI_INTENT.md) for examples and compatibility details.

## HTTP endpoints

When the API is running at http://127.0.0.1:8000:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Health check |
| POST | /retrieve | Retrieve relevant knowledge chunks |
| POST | /answer | Answer a question and maintain dialogue state |
| POST | /reset | Reset dialogue state |

## Documentation map

- [Full project guide](python/README.md) — installation, architecture, grammar, parser, semantic representation, retrieval, RAG, dialogue and evaluation.
- [Application/module guide](python/nlp/app/README.md) — module responsibilities, data schemas, CLI/API and evaluation workflows.
- [Grammar v5 multi-intent design](python/GRAMMAR_V5_MULTI_INTENT.md) — splitting rules, per-clause processing and output contract.
