# Hướng dẫn Cài đặt và Chạy Chương trình – Course Assistant (CO3085)

Tài liệu này hướng dẫn nhanh cách cài đặt môi trường và chạy toàn bộ các thành phần của hệ thống **Trợ lý hỗ trợ học tập thông minh (Course Assistant)**.

> Tài liệu mô tả chi tiết hệ thống theo mục nằm tại: [python/README.md](python/README.md).

---

## 1. Cấu trúc thư mục

```text
nlp/
├── README.md                  # Hướng dẫn cài đặt và chạy chương trình (file này)
└── python/
    ├── README.md              # Tài liệu đầy đủ 12 mục & cơ sở lý thuyết
    ├── Dockerfile             # Cấu hình đóng gói Docker
    ├── requirements.txt       # Các thư viện Python cần thiết
    ├── setup.py               # Cấu hình cài đặt package
    ├── data/                  # Knowledge Base môn học CO3085
    ├── nlp/app/               # Mã nguồn chính của hệ thống
    │   ├── input/             # Dữ liệu đầu vào kiểm thử
    │   └── output/            # Kết quả đầu ra cục bộ
    └── test/                  # Bộ kiểm thử tự động pytest
```

---

## 2. Hướng dẫn cài đặt

### 2.1 Yêu cầu môi trường
* **Python:** `>= 3.8` (khuyến nghị Python `3.10+` / `3.12`)
* **Hệ điều hành:** Windows / Linux / macOS
* **Docker:** (Tùy chọn) phục vụ đóng gói và chấm bài tự động

### 2.2 Cài đặt trực tiếp bằng `pip`
Mở terminal tại thư mục gốc `nlp/` và chạy:

```bash
cd python
pip install -r requirements.txt
pip install -e .
```

### 2.3 Cài đặt và chạy bằng Docker
Từ thư mục gốc `nlp/`:

```bash
# Build Docker image
docker build -t nlp-course-assistant python/

# Chạy container và mount thư mục input/output
docker run --rm -v "$(pwd)/output:/nlp/output" -v "$(pwd)/input:/nlp/input" nlp-course-assistant
```

---

## 3. Hướng dẫn chạy chương trình

Tất cả các lệnh dưới đây được thực thi từ thư mục `python/` (`cd python`):

### 3.1 Phần I: Văn phạm, Sinh câu và Phân tích cú pháp
```bash
# 1. Sinh câu mẫu hợp lệ từ văn phạm (tối đa 10.000 câu) -> nlp/app/output/samples.txt
python -m nlp.app.utils

# 2. Phân tích cú pháp tập câu đầu vào -> nlp/app/output/parse-results.txt
python -m nlp.app.run_parser \
  --input nlp/app/input/sentences.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt
```

### 3.2 Phần II, III & IV: Truy xuất, Hỏi đáp LLM/RAG, Hội thoại cục bộ và Đánh giá
```bash
# 1. Chạy truy xuất thông tin độc lập -> nlp/app/output/retrieval.txt
python -m nlp.app.run_retrieval

# 2. Chạy toàn bộ pipeline Course Assistant (Offline Mock LLM mặc định)
#    Đọc nlp/app/input/dialogues.txt và xuất dialogue.txt, answer.txt, evaluation.txt
#    ra cả nlp/app/output/ và ../output/
python -m nlp.app.run_assistant --data-dir data

# 3. Hỏi 1 câu đơn lẻ trên CLI (kèm xuất JSON nếu cần)
python -m nlp.app.run_assistant --data-dir data --question "Chương 6 nói về gì?"
python -m nlp.app.run_assistant --data-dir data --question "Phần II của bài tập lớn là gì?" --json

# 4. Chạy chế độ hội thoại tương tác nhiều lượt trên terminal (gõ 'reset' để xóa ngữ cảnh, 'exit' để thoát)
python -m nlp.app.run_assistant --data-dir data --interactive
```

### 3.3 Chạy với LLM thật qua API tương thích OpenAI (Tùy chọn)
```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m nlp.app.run_assistant --data-dir data --provider openai
```
*(Nếu không đặt `OPENAI_API_KEY` hoặc mất kết nối mạng, hệ thống tự động fallback về `MockLLMClient` để đảm bảo chạy offline 100%).*

### 3.4 Khởi động HTTP API Service
```bash
python -m nlp.app.api --host 127.0.0.1 --port 8000
```
* `GET http://127.0.0.1:8000/health`: Kiểm tra trạng thái server.
* `POST http://127.0.0.1:8000/retrieve`: Truy xuất top-k chunk với body `{"query": "Tuần 2 học những nội dung gì?"}`.
* `POST http://127.0.0.1:8000/answer`: Hỏi đáp có lưu ngữ cảnh hội thoại với body `{"question": "Chương 6 nói về gì?"}`.
* `POST http://127.0.0.1:8000/reset`: Xóa ngữ cảnh hội thoại hiện tại.

### 3.5 Chạy bộ kiểm thử tự động (`pytest`)
```bash
python -m pytest
```

### 3.6 Hướng dẫn chạy Local Model

```bash 
#sử dụng qwen2.5:7b cho demo này, có thể tải và sử dụng model khác, qwen có support cho tiếng việt
#từ thư mục root, tạo 1 terminal

$env:OLLAMA_HOST="0.0.0.0:11434"
$env:OLLAMA_DEBUG="1"                   # optional
$env:OLLAMA_DEBUG_LOG_REQUESTS="true"   # optional
ollama serve

#mở terminal thứ 2

export LLM_PROVIDER="api"
export LLM_MODEL="qwen2.5:7b"
export OPENAI_BASE_URL="http://172.23.64.1:11434/v1"
export OPENAI_API_KEY="ollama"
export LLM_TIMEOUT="120"
export PYTHONPATH="$PWD/python:$PYTHONPATH"

python -m nlp.app.run_assistant --data-dir data --provider openai
python -m nlp.app.api --host 127.0.0.1 --port 8000

#mở terminal thứ 3

curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"Tuần 2 học những nội dung gì?"}'

python3 python/run_evaluation.py --base-url http://127.0.0.1:8000 --dataset input/test_dataset.json --reset-before