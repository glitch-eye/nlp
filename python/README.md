# BÀI TẬP LỚN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (CO3085)
## Trợ lý hỗ trợ học tập thông minh – Course Assistant

---

## 1. Thông tin nhóm thực hiện
* **Môn học:** Xử lý ngôn ngữ tự nhiên (CO3085) - HK261
* **Đề tài:** Trợ lý hỗ trợ học tập thông minh (Course Assistant)
* **Thành viên nhóm:**
  - Sinh viên 1: [Họ và tên] - MSSV: [MSSV1]
  - Sinh viên 2: [Họ và tên] - MSSV: [MSSV2]
  - Sinh viên 3: [Họ và tên] - MSSV: [MSSV3]
  - Sinh viên 4: [Họ và tên] - MSSV: [MSSV4]

---

## 2. Hướng dẫn cài đặt và môi trường thực thi

### 2.1 Yêu cầu hệ thống
* Python `>= 3.8` (khuyến nghị Python 3.10+)
* Hệ điều hành: Linux / macOS / Windows
* Docker (dành cho chấm bài tự động và kiểm thử môi trường đóng gói)

### 2.2 Cài đặt thư viện
```bash
cd python
pip install -r requirements.txt
pip install -e .
```

### 2.3 Chạy qua Docker
Dự án cung cấp `Dockerfile` chuẩn theo yêu cầu bộ môn:
```bash
# Build docker image
docker build -t nlp-course-assistant python/

# Chạy test tự động với mount thư mục input/output
docker run --rm -v $(pwd)/output:/nlp/output -v $(pwd)/input:/nlp/input nlp-course-assistant
```
Hoặc dùng script có sẵn `util.sh`:
```bash
./util.sh test
./util.sh submit
```

---

## 3. Hướng dẫn chạy chương trình (Phần I)

### 3.0 API retrieval (Phần 1 RAG)
API này chỉ nhận query và trả các chunk bằng chứng được retriever chọn từ knowledge base. `samples_labeled.jsonl` dành cho đánh giá offline, không được dùng để chọn chunk khi API đang chạy.

Khởi động từ thư mục `sample-code-v1.0/python/`:
```bash
.venv/bin/python -m nlp.app.api --host 127.0.0.1 --port 8000
```

- `GET http://127.0.0.1:8000/health`: kiểm tra service.
- `POST http://127.0.0.1:8000/retrieve` với JSON `{"query":"Tuần 3 học gì?"}`: nhận `chunks`, `scores` và `retrieval_status`.

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Tuần 3 học gì?"}'
```

Retriever parse query theo CFG rồi kết hợp điểm semantic (intent/entity của query so với frame gắn vào từng chunk) với TF-IDF. Nếu parser không tạo được intent hợp lệ, hệ thống vẫn xếp hạng bằng TF-IDF. Mặc định trả tối đa 3 chunk (`top_k=3`), ngưỡng score `0.08`, trọng số TF-IDF `0.75`. API chưa sinh câu trả lời bằng LLM.

### 3.1 Sinh văn phạm (3.1)
File văn phạm chuẩn định dạng BNF được đặt tại:
* `output/grammar.txt` (hoặc `python/output/grammar.txt`, `python/hcmut/iaslab/nlp/app/output/grammar.txt`)

### 3.2 Sinh câu mẫu từ văn phạm (3.2)
Chạy giải thuật sinh câu (tối đa 10.000 câu hợp lệ thuộc miền bài toán):
```bash
python -m hcmut.iaslab.nlp.app.utils
```
Kết quả được xuất ra file: `output/samples.txt`.

### 3.3 Phân tích cú pháp câu hỏi (3.3)
Chạy bộ phân tích cú pháp (Earley Parser) trên tập câu kiểm thử `input/sentences.txt`:
```bash
python -m nlp.app.run_parser \
  --input nlp/app/input/sentences.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt

#with samples.txt
python -m nlp.app.run_parser \
  --input nlp/app/output/samples.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt
```
Đối với mỗi câu:
- Hợp lệ: xuất cây cú pháp dạng ngoặc `(S ...)`.
- Không hợp lệ / ngoài văn phạm: xuất `()`.

### 3.4 Chạy Course Assistant End-to-End (Phần II LLM/RAG & Phần III Hội thoại cục bộ)

Module điều phối [`assistant.py`](nlp/app/assistant.py), quản lý ngữ cảnh hội thoại [`dialogue.py`](nlp/app/dialogue.py) và CLI [`run_assistant.py`](nlp/app/run_assistant.py) kết nối toàn bộ pipeline:
`Dialogue Resolution -> CourseParser.parse() (Phần I) -> Semantic Frame -> Retriever (TF-IDF + Semantic) -> No-Context Guard -> Prompt -> LLMClient -> Citation Formatting -> DialogueState Update`.

#### a) Chế độ Offline / Mock LLM (Mặc định – Không cần mạng, không tốn chi phí)
Từ thư mục `python/`:
```bash
# 1. Chạy toàn bộ kịch bản hội thoại mẫu (input/dialogues.txt) và sinh output/dialogue.txt, answer.txt, evaluation.txt
python -m nlp.app.run_assistant

# 2. Hỏi 1 câu đơn lẻ trên CLI
python -m nlp.app.run_assistant --question "Chương 4 nói về gì?"

# 3. Xuất JSON đầy đủ schema chuẩn cho 1 câu hỏi
python -m nlp.app.run_assistant --question "Phần I cần xuất file gì?" --json

# 4. Chế độ hội thoại tương tác nhiều lượt (gõ 'reset' để xóa ngữ cảnh, 'exit' để thoát)
python -m nlp.app.run_assistant --interactive
```

#### b) Chế độ gọi Model thật qua API (`OpenAICompatibleLLMClient`)
Không commit API key vào mã nguồn. Cấu hình qua biến môi trường khi muốn gọi LLM thực:
```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"          # hoặc model tương thích OpenAI Chat Completions
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m nlp.app.run_assistant --provider openai
```
- **Giới hạn chi phí & kết nối:** Khi không đặt `OPENAI_API_KEY` hoặc khi mất kết nối mạng, hệ thống tự động chuyển về `MockLLMClient` (`mock-grounded-extractive-v1`) để đảm bảo tính tái lập 100% khi chấm bài offline.

#### c) Xem các file Trace & Output
Sau khi chạy `python -m nlp.app.run_assistant`, kiểm tra các file xuất tại `python/nlp/app/output/` (đồng thời được đồng bộ ra `output/` ở thư mục gốc):
- `output/dialogue.txt`: Trace chi tiết từng lượt hội thoại gồm 7 mục: (1) Câu gốc, (2) `DialogueState` trước, (3) `resolved_query` sau khi giải tham chiếu, (4) Các chunk truy xuất kèm `parse_status`, `retrieval_status`, `score`, (5) Câu trả lời và trạng thái (`ANSWERED`, `INSUFFICIENT_CONTEXT`, `NO_RELEVANT_CONTEXT`, `NEED_CLARIFICATION`), (6) Nguồn trích dẫn, (7) `DialogueState` sau.
- `output/answer.txt`: Danh sách câu trả lời chuẩn hóa kèm nguồn và chunk IDs.
- `output/assistant_run_summary.txt`: Thống kê của lần chạy hội thoại (Parser, Retriever, Answer Guard và Reference Resolution). Báo cáo đánh giá chuẩn `output/evaluation.txt` được tạo riêng bằng `python -m nlp.app.evaluation`.
- `output/retrieval.txt`: Trace truy xuất top-k chunk của `run_retrieval.py`.

#### d) Gọi qua HTTP API (`/retrieve`, `/answer`, `/reset`)
Selain endpoint `/retrieve`, service `nlp.app.api` hỗ trợ thêm:
- `POST http://127.0.0.1:8000/answer` với JSON `{"question": "Chương 4 nói về gì?"}` hoặc lượt tiếp theo `{"question": "Nó có liên quan đến parsing không?"}`.
- `POST http://127.0.0.1:8000/reset` để xóa `DialogueState` trên server.

---

## 4. Kiến trúc hệ thống (System Architecture)

Hệ thống được thiết kế theo pipeline xử lý ngôn ngữ tự nhiên:
```
Natural Language Query (Câu hỏi người dùng)
           │
           ▼
[ Earley Chart Parser ] ◄── [ output/grammar.txt ]
           │
           ▼
[ Syntactic Parse Tree ] ──► (Xuất kết quả Phần I: output/parse-results.txt)
           │
           ▼
[ Semantic Interpretation ] (Phần II: Ánh xạ cấu trúc cây sang biểu diễn logic)
           │
           ▼
[ Intent & Entity Extraction ] (Nhận diện Intent: GET_COURSE_INFO, GET_DEADLINE...)
           │
           ▼
[ Knowledge Base Query ] (Truy vấn dữ liệu môn học)
           │
           ▼
[ Natural Language Answer ] (Sinh câu trả lời hoặc thông báo khi không tìm thấy)
```

---

## 5. Mô tả chi tiết Văn phạm (Grammar) & Bộ phân tích cú pháp (Parser)

### 5.1 Văn phạm miền Course Assistant (`output/grammar.txt`)
Văn phạm thuộc lớp Ngữ pháp phi ngữ cảnh (CFG - Context-Free Grammar), mô tả các câu hỏi, yêu cầu và câu lịch sự trong miền Course Assistant, bao phủ 7 nhóm chức năng cốt lõi:
1. **`Q_COURSE_INFO`:** Hỏi thông tin chung của môn học (số tín chỉ, mã môn học, tên học phần...).
2. **`Q_TOPIC`:** Hỏi nội dung/chủ đề (tuần X/chương Y học gì, chủ đề Z học ở đâu/chương nào/tuần nào...).
3. **`Q_SCHEDULE`:** Hỏi lịch học và kế hoạch giảng dạy (học khi nào, học vào thứ mấy, lịch học diễn ra khi nào...).
4. **`Q_ASSIGNMENT`:** Hỏi thông tin bài tập, đồ án (bài tập lớn yêu cầu gì, gồm những phần nào, cần hoàn thành những bài tập nào...).
5. **`Q_DEADLINE`:** Hỏi thời hạn (deadline bài tập lớn là khi nào, hạn nộp khi nào...).
6. **`Q_RESOURCE`:** Hỏi tài liệu, slide, giáo trình (tài liệu nào nói về CFG, slide ở đâu, giáo trình môn học là gì...).
7. **`Q_RULE`:** Hỏi quy định môn học (quy định sử dụng AI, có được sử dụng AI không, quy định điểm danh...).
8. **`REQUEST`:** Các câu mệnh lệnh/yêu cầu tự nhiên (liệt kê tài liệu tham khảo, cho biết đề cương môn học...).
9. **`POLITE_QUERY`:** Mở rộng tiền tố lịch sự (cho hỏi..., xin hỏi..., vui lòng cho biết...).

### 5.2 Bộ phân tích cú pháp (Earley Parser)
* **Thuật toán:** Earley Chart Parsing.
* **Đặc điểm:**
  - Hỗ trợ phân tích đa từ (multi-word terminals) giúp nhận diện các cụm danh từ chuyên ngành như *"bài tập lớn"*, *"thời hạn nộp bài"*, *"Xử lý Ngôn ngữ Tự nhiên"*, *"CFG và Parsing"*.
  - Khử phân mảnh từ tùy tiện và chuẩn hóa Unicode (NFC), không phân biệt hoa thường (`casefold`).
  - Cache kết quả phân tích (`lru_cache`) tăng tốc độ xử lý hàng loạt.
  - Xử lý câu ngoài miền dữ liệu hoặc sai ngữ pháp: trả về `()` theo đúng yêu cầu đề bài.

---

## 6. Thiết kế Phần II: Biển diễn ngữ nghĩa & Hỏi đáp

1. **Biểu diễn ngữ nghĩa (Semantic Representation):**
   Cây cú pháp từ Phần I chứa các nút intent tương ứng ánh xạ trực tiếp:
   - `Q_COURSE_INFO` ➔ `GET_CREDIT(course=...)` hoặc `GET_COURSE_INFO(course=...)`
   - `Q_TOPIC` ➔ `GET_TOPIC(week=...)` hoặc `GET_TOPIC_LOCATION(topic=...)`
   - `Q_SCHEDULE` ➔ `GET_SCHEDULE(course=...)`
   - `Q_ASSIGNMENT` ➔ `GET_ASSIGNMENT(type=...)`
   - `Q_DEADLINE` ➔ `GET_DEADLINE(assignment=...)`
   - `Q_RESOURCE` ➔ `GET_RESOURCE(topic=...)`
   - `Q_RULE` ➔ `GET_RULE(rule_type=...)`

2. **Truy vấn Knowledge Base & Sinh câu trả lời:**
   - Dữ liệu môn học được lưu trữ có cấu trúc (danh mục môn học, lịch học, đề cương, quy định).
   - Nếu không tìm thấy thông tin: đưa ra thông báo chuẩn *"Xin lỗi, tôi không tìm thấy thông tin này trong dữ liệu của môn học."* thay vì sinh ảo (hallucination).

### 6.3 Nhánh LLM/RAG: model, prompt, answer guard và cách đánh giá

Luồng một lượt: `Parser (Phần I) -> Semantic Frame -> Retriever -> check_context -> build_prompt -> LLMClient.generate -> check_answer`.
LLM chỉ được gọi **sau** truy xuất và không thay thế parser/grammar.

**Model và cấu hình (`llm_client.py`)** - mọi cấu hình qua biến môi trường, không có khoá trong mã nguồn:

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `LLM_PROVIDER` hoặc `--provider` | `mock` / `openai` / `auto` (`auto` = dùng API nếu có khoá) | `mock` |
| `LLM_MODEL` hoặc `--model` | tên model (API tương thích OpenAI Chat Completions) | `gpt-4o-mini` |
| `OPENAI_API_KEY` | khoá API, không bao giờ được ghi ra output/log | - |
| `OPENAI_BASE_URL` | endpoint tương thích OpenAI | `https://api.openai.com/v1` |
| `LLM_TIMEOUT` | giây | `30` |

`model_name` ghi trong `output/answer.txt` và `output/evaluation.txt` luôn nêu **đúng nguồn của câu trả lời**:
`mock-grounded-extractive-v1` là mô hình giả lập chạy offline (trích văn bản từ chunk, **không phải LLM thật**);
nếu đã chọn API nhưng thiếu khoá hoặc mất kết nối, tên model có thêm `[FALLBACK: <lý do>, requested <model>]`.

**Prompt (`prompt_builder.py`)** gồm: luật hệ thống (chỉ dùng CONTEXT; không đoán ngày/yêu cầu/quy định;
không đủ thông tin thì trả đúng `INSUFFICIENT_CONTEXT`; trích dẫn `[Nguồn: file / section]` sao chép nguyên văn),
gợi ý ngữ nghĩa (ghi rõ *không phải bằng chứng*), câu hỏi và các khối CONTEXT, mỗi khối có nhãn nguồn.

**Answer guard (`answer_guard.py`)** gồm ba lớp:
1. *Trước LLM* (`check_context`): không có chunk phù hợp, hoặc câu không có intent/entity mà không từ nào của câu
   xuất hiện trong chunk -> `NO_RELEVANT_CONTEXT`, không gọi LLM.
2. *Sau LLM, sentinel*: model trả `INSUFFICIENT_CONTEXT` -> `INSUFFICIENT_CONTEXT`, không nguồn.
3. *Sau LLM, kiểm tra căn cứ* (`check_answer`): (a) trích dẫn phải khớp một chunk thực sự đưa vào prompt, trích dẫn
   bịa bị loại và câu trả lời thành `NEEDS_REVIEW`; (b) câu trả lời không có trích dẫn hợp lệ -> `NEEDS_REVIEW`;
   (c) số, ngày, giờ, e-mail, URL trong câu trả lời mà **không có** trong các chunk đã gửi -> câu trả lời bị giữ lại
   (`NEEDS_REVIEW`, bản gốc nằm ở `citation_check.raw_answer`). `sources` chỉ liệt kê nguồn được trích dẫn.

**Đánh giá tổng thể (`evaluation.py`)** là lệnh chuẩn tạo `output/evaluation.txt`. Báo cáo tích hợp Parser, Semantic,
Retrieval, End-to-End và tái sử dụng nguyên phần đánh giá RAG từ `rag_evaluation.py`: grounded answers, câu hỏi không có
thông tin/ngoài miền và các scripted hallucination probes kiểm tra Answer Guard. Scripted model chỉ là test double, không
được xem là kết quả của LLM thật.

```bash
# Báo cáo chuẩn Phần IV: evaluation.txt + evaluation_report.json
python -m nlp.app.evaluation --data-dir ../../NLP_Course_Assistant_Sample_Data

# Báo cáo RAG độc lập (nếu cần)
python -m nlp.app.rag_evaluation --data-dir ../../NLP_Course_Assistant_Sample_Data

# Chạy nhanh trên 100 mẫu
python -m nlp.app.evaluation --max-samples 100
```
Bộ đánh giá tổng thể mặc định dùng Mock LLM để không phát sinh chi phí API. Nếu dùng LLM thật cho demo/hỏi đáp,
hãy ghi rõ model và cấu hình; không nhầm số liệu Mock với chất lượng LLM thật.
Nếu không tìm thấy thư mục dữ liệu, Knowledge Base có thể rơi về trace fallback và số liệu retrieval không đại diện.

**Hạn chế đã biết:**
- Mock không đánh giá được độ liên quan: câu ngoài miền diễn đạt khác với mẫu của văn phạm (ví dụ "Thời tiết hôm nay thế nào?")
  có thể vẫn được trả lời bằng một chunk không liên quan. Với model thật, việc từ chối dựa vào sentinel `INSUFFICIENT_CONTEXT`.
- Kiểm tra trích dẫn chỉ xác nhận nhãn khớp chunk đã gửi, không chứng minh từng câu được chunk đó ủng hộ; kiểm tra thông tin chỉ bao phủ
  số/ngày/giờ/e-mail/URL, không bao phủ các khẳng định bằng lời (ví dụ tên người, "bắt buộc"/"không bắt buộc").
- Nhãn trong `samples_labeled.jsonl` được gán tự động (`AUTO_LABEL_REQUIRES_REVIEW`), chưa rà soát thủ công.

---

## 7. Đánh giá và hạn chế của hệ thống
* **Ưu điểm:**
  - Tốc độ xử lý cực nhanh, không phụ thuộc kết nối API hay tài nguyên tính toán lớn.
  - Phân tích cú pháp chính xác, cây cú pháp rõ ràng, dễ dàng kiểm soát lỗi.
  - Bộ sinh câu tạo ra 10.000 câu tự nhiên, thuần Việt, đúng ngữ cảnh môn học.
* **Hạn chế:**
  - Phụ thuộc vào tập luật văn phạm; các câu hỏi có cách diễn đạt quá sai lệch hoặc nhiều tiếng lóng sẽ bị từ chối (`()`).
  - Cần mở rộng thêm từ đồng nghĩa và từ vựng phong phú hơn khi mở rộng sang các môn học khác.
