# Course Assistant NLP package

Thư mục này chứa các module xử lý ngôn ngữ tự nhiên cho Course Assistant (CO3085).

## 1. Kiến trúc xử lý

Pipeline chính hiện tại (single-intent; chưa bao gồm multi-intent thử nghiệm):

```text
Natural Language Query
        |
        v
CourseParser / CFG
        |
        v
Semantic Frame (intent + entity IDs)
        |
        v
Knowledge Base + Retriever (TF-IDF / semantic signals)
        |
        v
Context / Answer Guard
        |
        v
Mock LLM hoặc OpenAI-compatible LLM
        |
        v
Grounded Answer + Citation
```

Các module chính:

| Module | Vai trò |
|---|---|
| `app/grammar.py`, grammar trong `app/output/` | Văn phạm miền Course Assistant |
| `app/parser.py` | Earley parser |
| `app/semantic_frame.py` | Ánh xạ parse tree sang intent và entity IDs |
| `app/knowledge_base.py` | Nạp dữ liệu môn học và chia thành chunks |
| `app/retriever.py` | Truy xuất top-k chunks |
| `app/assistant.py` | Điều phối pipeline hỏi đáp |
| `app/rag_evaluation.py` | Đánh giá RAG chuyên sâu và kiểm tra Answer Guard |
| `app/evaluation.py` | Bộ điều phối đánh giá tổng thể, tạo báo cáo chuẩn `evaluation.txt` |

## 2. Cài đặt

Yêu cầu Python 3.8 trở lên (khuyến nghị Python 3.10+). Thực hiện từ thư mục `python/` của repository:

```bash
pip install -r requirements.txt
pip install -e .
```

Đặt thư mục dữ liệu `NLP_Course_Assistant_Sample_Data` cạnh thư mục repository `nlp/` theo hướng dẫn trong README cấp trên, hoặc truyền đường dẫn dữ liệu bằng `--data-dir`.

## 3. Chạy Course Assistant

Từ thư mục `python/`:

```bash
# Chạy kịch bản hội thoại mẫu, sinh dialogue/answer và assistant_run_summary
python -m nlp.app.run_assistant

# Hỏi một câu
python -m nlp.app.run_assistant --question "Chương 4 nói về gì?"

# Chế độ tương tác nhiều lượt
python -m nlp.app.run_assistant --interactive
```

Mặc định hệ thống dùng Mock LLM chạy offline, không cần API key. Muốn gọi model tương thích OpenAI, cấu hình `OPENAI_API_KEY`, `LLM_MODEL`, `OPENAI_BASE_URL` và chạy với `--provider openai`. Khi dùng model thật, cần chú ý giới hạn chi phí.

## 4. Đánh giá tổng thể – tạo evaluation.txt

Đây là lệnh đánh giá chuẩn được khuyến nghị. Chạy từ thư mục `python/`:

```bash
python -m nlp.app.evaluation
```

Mặc định chương trình đọc:

```text
nlp/app/output/samples_labeled.jsonl
```

và tạo hai tệp trong `nlp/app/output/`:

- `evaluation.txt`: báo cáo tổng hợp dễ đọc, dùng làm output Phần IV.
- `evaluation_report.json`: số liệu chi tiết, lỗi theo từng mẫu, confusion counts và thông tin tái lập (dataset SHA-256, runtime, cấu hình).

Có thể chỉ định dữ liệu môn học và tham số:

```bash
python -m nlp.app.evaluation \
  --samples nlp/app/output/samples_labeled.jsonl \
  --data-dir ../../NLP_Course_Assistant_Sample_Data \
  --top-k 3 \
  --min-score 0.08
```

Chạy nhanh trên một tập con để kiểm tra pipeline:

```bash
python -m nlp.app.evaluation --max-samples 100
```

Bỏ qua một số tầng khi cần khoanh vùng lỗi:

```bash
# Bỏ đánh giá RAG end-to-end và các ví dụ hallucination
python -m nlp.app.evaluation --skip-e2e

# Chỉ đánh giá parser và semantic
python -m nlp.app.evaluation --skip-retrieval --skip-e2e
```

### Nội dung evaluation.txt

Báo cáo gồm:

1. **Parser / CFG:** parse-status accuracy, exact parse-tree match và số lỗi parser.
2. **Semantic:** intent accuracy, entity precision, recall và micro-F1.
3. **Retrieval:** Hit@k, UNKNOWN rejection và hành vi truy xuất trên mẫu INSUFFICIENT_CONTEXT. Chế độ đánh giá dùng query gốc, không đưa gold intent/entity vào retriever.
4. **End-to-end:** khả năng trả lời câu có context, từ chối câu thiếu/không có context, citation validity, NEEDS_REVIEW và over-answering.
5. **RAG evidence:** ví dụ grounded answer; câu hỏi không có dữ liệu/ngoài miền; các scripted hallucination probes để kiểm tra Answer Guard. Các scripted probes là test double, không phải kết quả của LLM thật.
6. **Dataset limitations / failure counts:** cảnh báo nhãn và số lỗi theo từng tầng.

Bộ đánh giá tổng thể dùng Mock LLM khi chạy CLI, để không phát sinh chi phí API và đảm bảo khả năng tái lập. Phần RAG evidence được tái sử dụng từ `rag_evaluation.py`, không thay thế hay loại bỏ bộ đánh giá RAG cũ.

### Đánh giá RAG riêng

Khi cần xem báo cáo RAG độc lập (không phải báo cáo tổng hợp), chạy:

```bash
python -m nlp.app.rag_evaluation \
  --data-dir ../../NLP_Course_Assistant_Sample_Data
```

Lệnh này tạo `nlp/app/output/rag_evaluation.txt`.

### Retrieval riêng

```bash
python -m nlp.app.run_retrieval \
  --labeled-jsonl nlp/app/output/samples_labeled.jsonl
```

Lệnh này tạo `retrieval.txt` và `retrieval_evaluation.json`.

## 5. Chạy test

Từ thư mục `python/`:

```bash
python -m unittest discover -s test -p "test_evaluation.py" -v
```

Lưu ý: chỉ xem kết quả đánh giá là đáng tin cậy sau khi rà soát nhãn. Dataset hiện có các dòng `AUTO_LABEL_REQUIRES_REVIEW`; nhãn intent, entity, parse tree và expected chunk cần được kiểm tra thủ công trước khi dùng số liệu làm kết luận học thuật.

## 6. Phân biệt các file output

| File | Ý nghĩa |
|---|---|
| `evaluation.txt` | Báo cáo đánh giá tổng thể chuẩn của `app.evaluation` |
| `evaluation_report.json` | Dữ liệu metrics chi tiết và lỗi theo mẫu |
| `rag_evaluation.txt` | Báo cáo RAG độc lập từ `app.rag_evaluation` |
| `retrieval.txt` | Trace retrieval độc lập |
| `assistant_run_summary.txt` | Thống kê của một lần chạy hội thoại từ `app.run_assistant` |
| `dialogue.txt`, `answer.txt` | Trace hội thoại và câu trả lời từ lần chạy assistant |

Các file output là sản phẩm sinh ra khi chạy; không cần commit kết quả chạy thử trừ khi nhóm muốn lưu một snapshot có ghi rõ dataset/config đã dùng.
