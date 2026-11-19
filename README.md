# Course Assistant NLP — CO3085

Trợ lý hỗ trợ học tập cho môn Xử lý Ngôn ngữ Tự nhiên (CO3085). Hệ thống gồm CFG/Earley Parser, Semantic Frame, Hybrid Retrieval, RAG/LLM, Answer Guard và quản lý hội thoại cục bộ. Nhánh hiện tại hỗ trợ cả single-intent và multi-intent.

- Hướng dẫn cài đặt và chạy nhanh: tài liệu này.
- Kiến trúc, module và hướng dẫn evaluation: [python/nlp/app/README.md](python/nlp/app/README.md).
- Tài liệu tổng quan và cơ sở lý thuyết: [python/README.md](python/README.md).
- Mô tả Grammar v5 multi-intent: [python/GRAMMAR_V5_MULTI_INTENT.md](python/GRAMMAR_V5_MULTI_INTENT.md).

## 1. Cấu trúc repository

~~~text
nlp/
├── README.md
├── evaluation/                 # Kết quả evaluation API (nếu có)
├── input/                      # Input cấp repository (nếu có)
├── output/                     # Output cấp repository
└── python/
    ├── README.md
    ├── GRAMMAR_V5_MULTI_INTENT.md
    ├── requirements.txt
    ├── setup.py
    ├── data/                   # Course knowledge base
    ├── nlp/app/
    │   ├── input/              # Dialogue và parser inputs
    │   └── output/             # Samples, traces và reports
    └── test/                   # unittest suite
~~~

## 2. Cài đặt

Yêu cầu Python 3.8 trở lên (khuyến nghị 3.10+). Từ thư mục repository:

~~~bash
cd python
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

**Docker:** Dockerfile hiện tại vẫn dùng entrypoint mẫu `hcommon.cli`, không khởi chạy Course Assistant. Dùng cách cài đặt pip ở trên; chưa xem Docker là phương thức chạy được hỗ trợ.

## 3. Chạy chương trình

Tất cả lệnh dưới đây chạy từ thư mục `python/`.

### 3.1 Grammar, sinh câu và Parser

~~~bash
python -m nlp.app.utils

python -m nlp.app.run_parser \
  --input nlp/app/input/sentences.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt
~~~

### 3.2 Retrieval và dataset

~~~bash
# Retrieval độc lập
python -m nlp.app.run_retrieval

# Sinh samples và nhãn tự động từ grammar + course data
python -m nlp.app.build_sample_dataset --max-sentences 10000

# Tạo hàng đợi để con người kiểm tra nhãn
python -m nlp.app.label_audit \
  --input nlp/app/output/samples_labeled.jsonl \
  --output nlp/app/output/label_audit_queue.jsonl \
  --per-stratum 2 \
  --max-samples 300

# Export riêng những hàng đã review hợp lệ
python -m nlp.app.label_audit \
  --input nlp/app/output/samples_labeled.jsonl \
  --review-queue nlp/app/output/label_audit_queue.jsonl \
  --apply-reviewed \
  --reviewed-output nlp/app/output/samples_reviewed.jsonl
~~~

Nhãn do builder sinh tự động không phải human-reviewed gold labels. Chỉ dùng `samples_reviewed.jsonl` sau khi reviewer thực sự kiểm tra và điền kết quả. Xem [hướng dẫn label audit và evaluation](python/nlp/app/README.md#9-sinh-dataset-có-nhãn-cho-evaluation).

### 3.3 Course Assistant

~~~bash
# Batch dialogue mặc định, Mock LLM offline
python -m nlp.app.run_assistant --data-dir data

# Một câu hỏi
python -m nlp.app.run_assistant --data-dir data \
  --question "Chương 6 nói về gì?" --json

# Hội thoại tương tác
python -m nlp.app.run_assistant --data-dir data --interactive
~~~

### 3.4 Chạy với API model thật

~~~bash
export OPENAI_API_KEY="YOUR_API_KEY"
export LLM_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m nlp.app.run_assistant --data-dir data --provider openai
~~~

Nếu API key thiếu hoặc request thất bại, client có thể fallback về Mock. Chọn `--provider openai` không tự chứng minh model thật đã sinh câu trả lời; dùng answer/unified evaluation và kiểm tra telemetry. Không commit API key.

### 3.5 HTTP API

~~~bash
python -m nlp.app.api --host 127.0.0.1 --port 8000
~~~

Các endpoint:
- `GET /health`: health check và thông tin retriever.
- `POST /retrieve`: body `{"query":"Tuần 3 học gì?"}`.
- `POST /answer`: body `{"question":"Chương 4 nói về gì?"}`.
- `POST /reset`: reset dialogue state.

API server dùng Mock theo mặc định. Muốn API `/answer` gọi model thật, cấu hình môi trường trước khi khởi động server:

~~~bash
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="YOUR_API_KEY"
export LLM_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
python -m nlp.app.api --host 127.0.0.1 --port 8000
~~~

### 3.6 Local model qua Ollama (tùy chọn)

Cài Ollama và tải model, ví dụ:

~~~bash
ollama pull qwen2.5:7b
ollama serve
~~~

Trong terminal khác, từ thư mục `python/`:

~~~bash
export LLM_PROVIDER="api"
export LLM_MODEL="qwen2.5:7b"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
export LLM_TIMEOUT="120"

python -m nlp.app.run_assistant --data-dir data --provider openai
~~~

Nếu Assistant và Ollama chạy trên hai máy khác nhau, thay `localhost` bằng địa chỉ host mà máy Assistant truy cập được. Không dùng IP cố định từ ví dụ của một máy cụ thể.

Để API server dùng Ollama, giữ các biến môi trường trên rồi chạy:

~~~bash
python -m nlp.app.api --host 127.0.0.1 --port 8000
~~~

### 3.7 Evaluation

~~~bash
# Parser / semantic / retrieval / end-to-end evaluation
python -m nlp.app.evaluation

# RAG evaluation độc lập
python -m nlp.app.rag_evaluation --provider mock

# Answer-generation evaluation (offline)
python -m nlp.app.answer_evaluation --provider mock

# Unified report (offline)
python -m nlp.app.unified_evaluation --provider mock --max-samples 10
~~~

Đánh giá bằng model thật:

~~~bash
export OPENAI_API_KEY="YOUR_API_KEY"
python -m nlp.app.unified_evaluation \
  --provider openai --require-real-model --max-samples 5
~~~

Dataset mặc định và output path được mô tả trong [README package](python/nlp/app/README.md#10-evaluation-tổng-thể-và-rag-evaluation). Không gộp kết quả Mock, scripted probes và model thật. Multi-intent hiện chạy per-clause; một query có thể phát sinh nhiều generation calls.

### 3.8 Test

~~~bash
python -m unittest discover -s test -v
~~~

## 4. Lưu ý về kết quả

- Grammar-generated samples hữu ích cho regression và coverage, không đại diện toàn bộ câu hỏi tự nhiên.
- Auto-label không đồng nghĩa với human-reviewed label.
- `--use-gold-entities` là chế độ oracle/upper-bound, không phải query-only retrieval.
- Answer Guard giảm một số dạng unsupported output nhưng không bảo đảm loại bỏ hoàn toàn hallucination.
- Multi-intent hiện tách và xử lý từng clause; chưa gom toàn bộ intent thành một lần gọi LLM duy nhất.
