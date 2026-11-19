# Course Assistant NLP — CO3085

Hệ thống trợ lý hỗ trợ học tập cho môn **Xử lý Ngôn ngữ Tự nhiên (CO3085)**. Repository hiện thực pipeline CFG/Earley Parser, Semantic Frame, Hybrid Retrieval, RAG/LLM, Answer Guard và quản lý hội thoại cục bộ.

- Hướng dẫn cài đặt và chạy nhanh: tài liệu này.
- Kiến trúc, cơ sở lý thuyết và mô tả module: [python/README.md](python/README.md).

## 1. Cấu trúc repository

~~~text
nlp/
├── README.md
└── python/
    ├── README.md
    ├── Dockerfile             # Hiện chưa cấu hình để chạy Course Assistant
    ├── requirements.txt
    ├── setup.py
    ├── data/                  # Course knowledge base + entities.txt
    ├── nlp/app/
    │   ├── input/             # Dialogue và query inputs
    │   └── output/            # Generated samples, traces và reports
    └── test/                  # unittest suite
~~~

## 2. Yêu cầu và cài đặt

- Python 3.8+ (khuyến nghị Python 3.10 hoặc 3.12).
- Windows, macOS hoặc Linux.
- API key chỉ cần khi muốn gọi model thật qua OpenAI-compatible API.

Từ thư mục repository:

~~~bash
cd python
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

Knowledge base mặc định nằm tại `python/data/`. Có thể truyền đường dẫn khác bằng `--data-dir`.

### Docker

**Chưa sử dụng các lệnh Docker bên dưới như một cách chạy Course Assistant.** Dockerfile hiện tại vẫn dùng entrypoint mẫu `hcommon.cli`, không khởi chạy `nlp.app`; các volume `input/output` trước đây trong README cũng không khớp với đường dẫn dữ liệu của ứng dụng. Cần cập nhật và kiểm thử Dockerfile riêng trước khi công bố Docker là phương thức cài đặt được hỗ trợ.

## 3. Hướng dẫn chạy

Tất cả lệnh trong phần này được chạy từ thư mục `python/`.

### 3.1 Grammar, sinh câu và Parser

~~~bash
# Sinh tối đa 10.000 câu từ grammar
python -m nlp.app.utils

# Parse tập câu đầu vào
python -m nlp.app.run_parser \
  --input nlp/app/input/sentences.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt
~~~

### 3.2 Retrieval và tạo tập dữ liệu đánh giá

~~~bash
# Chạy retrieval trên data/sample_queries.txt
python -m nlp.app.run_retrieval

# Sinh tập synthetic samples_labeled.jsonl (đầu vào cho bước kế tiếp)
python -m nlp.app.build_sample_dataset

# Tạo/cập nhật tập nhãn cho retrieval/RAG evaluation
python -m nlp.app.build_gold_labels
~~~

Lệnh tạo dữ liệu mặc định sinh:
- `nlp/app/input/test_labeled.jsonl`: 114 câu hỏi được tuyển chọn riêng làm tập kiểm thử.
- `nlp/app/output/samples_gold_labeled.jsonl`: các mẫu synthetic từ `samples_labeled.jsonl` được gán lại expected entities, chunks và answerability.

**Lưu ý về nhãn:** script hiện tự động tạo nhãn bằng quy tắc/parser/knowledge base và gán trường `label_status="GOLD_REVIEWED"`. Giá trị này không chứng minh rằng con người đã review từng dòng. Trước khi báo cáo đây là *human-reviewed gold set*, cần kiểm tra nhãn độc lập; đặc biệt không xem tập synthetic 8.638 câu là test set độc lập.

Đánh giá retrieval trên tập 114 câu:

~~~bash
python -m nlp.app.run_retrieval \
  --labeled-jsonl nlp/app/input/test_labeled.jsonl \
  --output nlp/app/output/retrieval_test.txt \
  --evaluation-output nlp/app/output/retrieval_test_eval.json
~~~

Đánh giá retrieval trên tập synthetic được gán lại nhãn:

~~~bash
python -m nlp.app.run_retrieval \
  --labeled-jsonl nlp/app/output/samples_gold_labeled.jsonl \
  --output nlp/app/output/retrieval_synthetic.txt \
  --evaluation-output nlp/app/output/retrieval_synthetic_eval.json
~~~

Mặc định retriever dùng intent/entity do pipeline phân tích câu hỏi tạo ra. Tùy chọn `--use-gold-entities` đưa intent/entity nhãn vào retrieval như một chế độ oracle/upper-bound; không dùng kết quả này thay cho kết quả retrieval query-only.

### 3.3 RAG evaluation

~~~bash
# Mock/offline trên tập 114 câu
python -m nlp.app.rag_evaluation \
  --samples nlp/app/input/test_labeled.jsonl \
  --output-dir nlp/app/output/rag_test

# Mock/offline trên tập synthetic
python -m nlp.app.rag_evaluation \
  --samples nlp/app/output/samples_gold_labeled.jsonl \
  --output-dir nlp/app/output/rag_synthetic
~~~

Mỗi lệnh tạo `rag_evaluation.txt` trong thư mục output tương ứng. Dùng thư mục riêng để tránh lần chạy sau ghi đè báo cáo trước. Với model thật, đặt `OPENAI_API_KEY` và thêm `--provider openai --max-samples N`; kiểm tra model/provenance trong report vì API failure có thể fallback về Mock.

### 3.4 Course Assistant và hội thoại

~~~bash
# Chạy dialogue script mặc định, Mock LLM offline
python -m nlp.app.run_assistant --data-dir data

# Một câu hỏi
python -m nlp.app.run_assistant --data-dir data \
  --question "Chương 6 nói về gì?" --json

# Hội thoại tương tác; dùng reset để xóa context, exit để thoát
python -m nlp.app.run_assistant --data-dir data --interactive
~~~

Các output chính gồm `dialogue.txt`, `answer.txt`, `evaluation.txt` và `rag_evaluation.txt` trong `nlp/app/output/`. Runner cũng mirror một số output sang thư mục `output/` ở repository root.

### 3.5 Dùng OpenAI-compatible API

~~~bash
export OPENAI_API_KEY="YOUR_API_KEY"
export LLM_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m nlp.app.run_assistant --data-dir data --provider openai
~~~

Không commit API key. Nếu thiếu key hoặc request lỗi, `LLMClient` có thể fallback sang Mock. Vì vậy, chọn `--provider openai` không tự chứng minh câu trả lời được sinh bởi model thật; cần kiểm tra model name/fallback marker trong output.

### 3.6 Chạy với Ollama (tùy chọn)

Cài Ollama và tải model, ví dụ:

~~~bash
ollama pull qwen2.5:7b
ollama serve
~~~

Mở terminal khác, từ thư mục `python/`:

~~~bash
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="qwen2.5:7b"
export LLM_TIMEOUT="120"

python -m nlp.app.run_assistant --data-dir data --provider openai
~~~

Nếu Course Assistant và Ollama chạy trên hai máy hoặc trong hai môi trường mạng khác nhau, thay `localhost` bằng địa chỉ host mà máy chạy Assistant có thể truy cập được. Không dùng một IP cố định trong tài liệu vì địa chỉ phụ thuộc môi trường.

Để chạy RAG evaluation bằng Ollama:

~~~bash
python -m nlp.app.rag_evaluation \
  --samples nlp/app/input/test_labeled.jsonl \
  --provider openai \
  --model qwen2.5:7b \
  --max-samples 20 \
  --output-dir nlp/app/output/rag_ollama
~~~

### 3.7 HTTP API

~~~bash
python -m nlp.app.api --host 127.0.0.1 --port 8000
~~~

- `GET /health`: health check.
- `POST /retrieve`: truy xuất chunks, ví dụ body `{"query":"Tuần 2 học những nội dung gì?"}`.
- `POST /answer`: hỏi đáp, ví dụ body `{"question":"Chương 6 nói về gì?"}`.
- `POST /reset`: reset dialogue state.

### 3.8 Kiểm thử tự động

Các test hiện dùng `unittest`:

~~~bash
python -m unittest discover -s test -v
~~~

Có thể dùng `pytest` nếu đã cài riêng, nhưng `pytest` không nằm trong `requirements.txt` hiện tại.

## 4. Tài liệu liên quan

- [Tài liệu kỹ thuật đầy đủ](python/README.md) — kiến trúc, Grammar, Parser, Semantic Frame, Retrieval, RAG, hội thoại và đánh giá.
- [Source code ứng dụng](python/nlp/app/).
- [Test suite](python/test/).
