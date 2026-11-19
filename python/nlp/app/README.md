# Course Assistant NLP — `nlp.app`

Tài liệu kỹ thuật và hướng dẫn chạy cho toàn bộ package `nlp/app` của đề tài Course Assistant (CO3085). Đây là **README duy nhất dành cho package `nlp.app`**: kiến trúc, các module, grammar/parser, sinh dữ liệu, semantic, retrieval, RAG, hội thoại, API và evaluation được tập trung tại đây.

Phạm vi hiện tại hỗ trợ cả single-intent và multi-intent. Single-intent giữ nguyên semantic-frame contract; multi-intent dùng `intent: MULTI_INTENT` cùng `clauses` riêng cho từng intent.

## 1. Tổng quan pipeline

### 1.1 Luồng xử lý một câu hỏi

```text
User question
     |
     v
DialogueManager
(resolve references from previous turn, if any)
     |
     v
CourseParser (Earley) <----- output/grammar.txt
     |
     v
Semantic Frame
(intent + canonical entity IDs)
     |
     v
Knowledge Base + Retriever
(TF-IDF lexical score + semantic intent/entity score)
     |
     v
Answer Guard: check_context
     |
     v
Prompt Builder
     |
     v
LLMClient (Mock by default / OpenAI-compatible API)
     |
     v
Answer Guard: check_answer
     |
     v
Answer + citations + status
     |
     v
DialogueState update
```

Grammar/parser/semantic cung cấp cấu trúc và gợi ý truy xuất. **Các chunk lấy từ Knowledge Base mới là bằng chứng để trả lời**; intent/entity không được xem là bằng chứng nội dung.

### 1.2 Các trạng thái đầu ra chính

| Trạng thái | Ý nghĩa |
|---|---|
| `ANSWERED` | Có câu trả lời và citation hợp lệ theo các chunk đã gửi vào prompt |
| `INSUFFICIENT_CONTEXT` | Chunk không đủ thông tin; hệ thống từ chối suy đoán |
| `NO_RELEVANT_CONTEXT` | Không tìm được bằng chứng liên quan trong Knowledge Base |
| `NEED_CLARIFICATION` | Cần người dùng làm rõ tham chiếu hoặc lựa chọn |
| `NEEDS_REVIEW` | Câu trả lời có citation không hợp lệ, thiếu citation hoặc chứa facts không được chunk hỗ trợ |

## 2. Cấu trúc thư mục

Các đường dẫn dưới đây tính từ thư mục `python/`:

```text
nlp/
└── app/
    ├── README.md                 # Tài liệu package này
    ├── input/
    │   ├── dialogues.txt         # Kịch bản hội thoại nhiều lượt
    │   └── sentences.txt         # Câu đầu vào cho parser
    ├── output/
    │   ├── grammar.txt           # CFG nguồn đang được parser sử dụng
    │   ├── samples.txt           # Câu được sinh từ grammar
    │   ├── samples_labeled.jsonl # Dataset có nhãn cho evaluation
    │   ├── parse-results.txt     # Kết quả parser
    │   ├── dialogue.txt          # Trace hội thoại
    │   ├── answer.txt            # Câu trả lời và nguồn
    │   ├── assistant_run_summary.txt
    │   ├── evaluation.txt        # Báo cáo evaluation tổng thể
    │   ├── evaluation_report.json
    │   ├── rag_evaluation.txt    # Báo cáo RAG độc lập
    │   └── retrieval.txt         # Trace retrieval
    ├── answer_guard.py
    ├── api.py
    ├── assistant.py
    ├── build_sample_dataset.py
    ├── course_grammar.py
    ├── dialogue.py
    ├── evaluation.py
    ├── grammar_file.py
    ├── knowledge_base.py
    ├── llm_client.py
    ├── parser.py
    ├── prompt_builder.py
    ├── rag_evaluation.py
    ├── retriever.py
    ├── run_assistant.py
    ├── run_parser.py
    ├── run_retrieval.py
    ├── semantic.py
    ├── semantic_frame.py
    ├── sentence_generator.py
    ├── text_utils.py
    └── utils.py
```

Một số file JSON/trace khác có thể xuất hiện trong `output/` tùy các lần chạy retrieval/evaluation. Đây là thư mục dữ liệu đầu vào và kết quả chạy, không phải nơi chứa mã nguồn.

## 3. Vai trò các module

| Module | Trách nhiệm |
|---|---|
| `grammar_file.py` | Đọc CFG dạng text từ `output/grammar.txt`, kiểm tra production và ký hiệu |
| `course_grammar.py` | Adapter cung cấp grammar cho bộ sinh câu |
| `sentence_generator.py` | Sinh các câu từ production bằng duyệt breadth-first, loại câu trùng, giới hạn số câu và độ sâu mở rộng |
| `utils.py` | CLI sinh `output/samples.txt` |
| `parser.py` | Earley chart parser, tạo `ParseNode` và cây cú pháp dạng ngoặc |
| `run_parser.py` | CLI parse file câu hỏi và ghi kết quả |
| `semantic_frame.py` | Ánh xạ parse tree/query/chunk thành intent và entity IDs chuẩn |
| `semantic.py` | Adapter semantic: bổ sung alias từ `entities.txt`, category-keyed entities và trạng thái PARSED/UNPARSED |
| `text_utils.py` | Chuẩn hóa văn bản và tạo slug dùng chung |
| `knowledge_base.py` | Đọc dữ liệu môn học, chia tài liệu thành chunks có source/section/entity metadata |
| `retriever.py` | Xếp hạng chunks bằng TF-IDF kết hợp semantic intent/entity |
| `dialogue.py` | Lưu DialogueState và giải tham chiếu cục bộ giữa các lượt |
| `prompt_builder.py` | Tạo prompt yêu cầu trả lời chỉ dựa trên context, theo format citation thống nhất |
| `llm_client.py` | Adapter LLM: Mock offline hoặc API tương thích OpenAI |
| `answer_guard.py` | Chặn câu không có context, sentinel thiếu thông tin, citation sai và facts số/ngày/giờ/email/URL không có trong chunks |
| `assistant.py` | Điều phối pipeline end-to-end |
| `run_assistant.py` | CLI hỏi đơn, chạy hội thoại, interactive mode và ghi trace |
| `api.py` | HTTP API cho retrieval, answer và reset dialogue |
| `run_retrieval.py` | Chạy thử và đánh giá retrieval độc lập |
| `build_sample_dataset.py` | Sinh dataset JSONL có parse/semantic labels và expected chunk IDs |
| `evaluation.py` | Đánh giá Parser, Semantic (single/multi-intent), Retrieval, End-to-End và RAG evidence |
| `rag_evaluation.py` | Đánh giá RAG chuyên sâu độc lập, gồm grounded/no-info examples và scripted Answer Guard probes |

## Nguồn dữ liệu Course Assistant

Các chương trình Assistant, Retrieval, RAG Evaluation và bộ sinh dataset mặc định đọc dữ liệu nội bộ tại `python/data/`:

- `course_info.txt`, `schedule.txt`, `topics.txt`
- `assignments.txt`, `resources.txt`, `regulations.txt`, `faq.txt`
- `entities.txt`, `sample_queries.txt`

Không còn fallback dựng Knowledge Base từ `output/retrieval.txt`. Nếu thiếu dữ liệu bắt buộc, chương trình dừng và báo file còn thiếu thay vì âm thầm chạy trên dữ liệu một phần. Có thể truyền `--data-dir` để chủ động chọn một bộ dữ liệu khác có cùng cấu trúc.

Chạy từ thư mục `python/`:

```bash
python -m nlp.app.run_assistant
python -m nlp.app.run_retrieval
python -m nlp.app.build_sample_dataset --max-sentences 10000
python -m nlp.app.rag_evaluation --provider mock
python -m nlp.app.evaluation
```

## 4. Grammar, sentence generation và parser

### 4.1 Grammar là nguồn luật duy nhất

Grammar đang dùng mặc định:

```text
nlp/app/output/grammar.txt
```

Đây là CFG dạng text. `grammar_file.load_grammar()` đọc nonterminal dạng `<NAME>`, terminal phải đặt trong dấu ngoặc kép, các alternative phân tách bằng `|`. Start rule bắt buộc là `<S>`.

Các nhánh gốc `Q_*` trong grammar được dùng để suy ra intent cho bộ dataset. Grammar được giới hạn theo các dạng câu hỏi thuộc miền Course Assistant và sample knowledge base; việc sinh được câu và parse thành công không chứng minh câu tự nhiên hoặc bao phủ mọi cách hỏi thực tế.

### 4.2 Sinh câu từ grammar

Chạy từ thư mục `python/`:

```bash
python -m nlp.app.utils
```

Mặc định tạo tối đa 10.000 câu tại `nlp/app/output/samples.txt`.

Tùy chỉnh:

```bash
python -m nlp.app.utils \
  --output nlp/app/output/samples.txt \
  --max-sentences 5000
```

`SentenceGenerator` duyệt các derivation theo breadth-first, loại câu trùng và giới hạn mở rộng để tránh vòng lặp/đệ quy vô hạn. Giới hạn tối đa hiện tại là 10.000 câu.

### 4.3 Parse câu hỏi

```bash
# Dùng mặc định input/sentences.txt và output/grammar.txt
python -m nlp.app.run_parser

# Parse file samples được sinh từ grammar
python -m nlp.app.run_parser \
  --input nlp/app/output/samples.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt
```

Parser dùng Earley chart parsing. Parser chuẩn hóa Unicode NFC, hỗ trợ terminal nhiều từ và cache kết quả parse. Câu hợp lệ tạo parse tree; câu không khớp grammar trả về `None` (CLI ghi dạng cây rỗng `()`).

Các tùy chọn: `--input`, `--output`, `--grammar`.

## 5. Semantic frame và entity

### Single-intent và Multi-intent

- Câu đơn tiếp tục dùng `parser.parse()` và semantic frame dạng `{intent, entity_ids, parse_status}`; không thêm trường `clauses` vào kết quả single-intent.
- Câu đa ý định chỉ được nhận diện khi `parse_multi()` tạo cây `S_MULTI` sau khi từng clause được kiểm tra độc lập bằng CFG.
- Multi-intent trả `intent: MULTI_INTENT`, `clauses: [...]` và `entity_ids` tổng hợp. Mỗi clause giữ `intent`, `entity_ids`, `parse_status` riêng.
- Evaluation ưu tiên `parse_multi()` nếu parser hỗ trợ, nhưng vẫn fallback về `parse()` cho parser cũ. Gold labels multi-intent có thể lưu `clauses` hoặc `intents`; intent accuracy dùng exact-set match giữa các intent của clause.
- Không gán `MULTI_INTENT` cho câu chỉ vì có từ “và”. Nếu không tách được thành các clause độc lập parse hợp lệ, câu vẫn đi theo single-intent CFG.


Sau khi parse, `semantic_frame.py`/ `semantic.py` tạo biểu diễn gồm:

- `intent`: nhóm yêu cầu, ví dụ `GET_TOPIC`, `GET_SCHEDULE`, `GET_ASSIGNMENT`, `GET_RESOURCE`, `GET_RULE`.
- `entity_ids`: các định danh chuẩn như `CH04`, `WEEK_05`, `BTL01`, `LO2.1`.
- `entities`: entity theo category để DialogueState sử dụng, ví dụ `chapter -> CH04`.
- `parse_status`: `PARSED` hoặc `UNPARSED`.

Alias và canonical entity values được nạp từ `entities.txt` trong sample data. Semantic frame là tín hiệu giúp retriever định hướng; nó không thay thế nội dung chunk.

Nếu parser không tạo được cây, semantic adapter trả frame `UNPARSED` mà không tự đoán intent.

## 6. Knowledge Base và Retrieval

### 6.1 Dữ liệu đầu vào

Knowledge Base mặc định tìm thư mục `NLP_Course_Assistant_Sample_Data` theo layout repository. Có thể truyền đường dẫn tường minh bằng `--data-dir`.

Các file dữ liệu được `knowledge_base.py` nhận diện:

| File | Nhóm nội dung |
|---|---|
| `course_info.txt` | Thông tin chung môn học |
| `schedule.txt` | Lịch học, chia theo WEEK |
| `topics.txt` | Nội dung, chia theo CHAPTER |
| `assignments.txt` | Bài tập/đồ án |
| `resources.txt` | Tài liệu tham khảo |
| `regulations.txt` | Quy định |
| `faq.txt` | Câu hỏi thường gặp |
| `entities.txt` | Alias và canonical entity mapping |

Mỗi chunk lưu `chunk_id`, `text`, `source`, `section`, `entity_ids` và `kind`. Nhờ metadata này, citation có thể chỉ đến file và section cụ thể.

### 6.2 Cách xếp hạng

Retriever dùng TF-IDF cosine similarity cho tín hiệu lexical và semantic score từ intent/entity frame.

Cấu hình mặc định trong code:

| Tham số | Mặc định |
|---|---:|
| `top_k` | 3 |
| `min_score` | 0.08 |
| `lexical_weight` | 0.75 |
| `semantic_weight` | 0.25 |
| `min_lexical_score` | 0.0 |

Khi query có intent hợp lệ, điểm tổng hợp kết hợp lexical và semantic score. Khi không có semantic intent hợp lệ, retriever dùng lexical score. Intent `UNKNOWN` bị bỏ qua, không truy xuất chunk.

### 6.3 Chạy retrieval độc lập

```bash
python -m nlp.app.run_retrieval \
  --labeled-jsonl nlp/app/output/samples_labeled.jsonl
```

Mặc định ghi:

- `nlp/app/output/retrieval.txt`: trace xếp hạng.
- `nlp/app/output/retrieval_evaluation.json`: metrics và kết quả theo mẫu.

Nếu không truyền `--labeled-jsonl`, chương trình đọc `sample_queries.txt` trong data directory và chỉ ghi trace retrieval.

Các tham số chính: `--data-dir`, `--queries`, `--labeled-jsonl`, `--output`, `--evaluation-output`, `--top-k`, `--min-score`, `--min-lexical-score`.

`--use-gold-entities` chỉ dành cho chế độ oracle/upper-bound để đo retrieval khi cung cấp intent/entity labels đã biết. Không dùng chế độ này để báo cáo kết quả retrieval thực tế vì nó đưa nhãn đáp án vào đường truy xuất.

## 7. Course Assistant, hội thoại và Answer Guard

### 7.1 Chạy Assistant

Từ thư mục `python/`:

```bash
# Chạy kịch bản nhiều lượt mặc định: input/dialogues.txt
python -m nlp.app.run_assistant

# Hỏi một câu
python -m nlp.app.run_assistant \
  --question "Chương 4 nói về gì?"

# In kết quả theo JSON
python -m nlp.app.run_assistant \
  --question "Phần I cần xuất file gì?" --json

# Hội thoại tương tác; gõ reset để xóa ngữ cảnh, exit để thoát
python -m nlp.app.run_assistant --interactive
```

Một số tùy chọn:

- `--dialogue-file PATH`: file kịch bản nhiều lượt.
- `--input PATH`: file câu hỏi một lượt, mỗi dòng một câu.
- `--data-dir PATH`: thư mục dữ liệu môn học.
- `--grammar PATH`: grammar override.
- `--provider mock|openai|auto`: chọn LLM provider.
- `--model NAME`: model API.
- `--top-k N`: số chunk truy xuất.
- `--output-dir DIR`: thư mục ghi trace.
- `--eval-samples N`: số mẫu nhãn dùng cho phần RAG evidence của run.
- `--reset`: reset state trước câu hỏi đơn.

### 7.2 Dialogue state

`DialogueState` lưu câu hỏi gần nhất, intent, entities, answer entities và sources. `DialogueManager` dùng các giá trị này để giải tham chiếu như “chương này”, “tuần này”, “phần này”, “nó”, “chương sau” trước khi truy xuất.

Đây là xử lý ngữ cảnh hội thoại cục bộ theo các pattern đã định nghĩa; không phải bộ giải quyết đồng tham chiếu tổng quát cho mọi diễn đạt. Multi-intent/multi-clause tổng quát chưa nằm trong phạm vi hiện tại.

### 7.3 LLM và prompt

Mặc định Assistant dùng Mock LLM chạy offline. Có thể cấu hình API tương thích OpenAI qua biến môi trường:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export LLM_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python -m nlp.app.run_assistant --provider openai
```

Không commit API key vào repository. Provider `auto` chọn API khi có credential phù hợp; nếu muốn đánh giá tái lập/offline, dùng `mock`.

Prompt yêu cầu model chỉ dựa vào CONTEXT được truy xuất, không tự đoán thông tin môn học và dùng citation theo định dạng:

```text
[Nguồn: source / section]
```

### 7.4 Answer Guard

- **Trước LLM – `check_context`:** chặn trường hợp retriever không tìm thấy chunk phù hợp; với query không có intent/entity, dùng lexical support thận trọng để phát hiện thiếu liên quan.
- **Sau LLM – sentinel:** nếu model trả `INSUFFICIENT_CONTEXT`, chuyển thành trạng thái thiếu bằng chứng và không đưa nguồn.
- **Sau LLM – citation:** citation phải trỏ đến chunk thực sự được gửi trong prompt; citation không hợp lệ hoặc không có citation sẽ chuyển sang `NEEDS_REVIEW`.
- **Fact guard:** số, ngày, giờ, email và URL trong câu trả lời phải xuất hiện trong nội dung/section của chunk đã gửi; facts không được hỗ trợ sẽ bị giữ lại và chuyển `NEEDS_REVIEW`.

Giới hạn: citation hợp lệ chỉ xác nhận nguồn được trích nằm trong context đã gửi, không chứng minh mọi mệnh đề ngôn ngữ tự nhiên trong câu trả lời đều được nguồn hỗ trợ. Fact guard hiện tập trung vào các dạng facts có thể trích xuất bằng regex, không phải kiểm chứng ngữ nghĩa đầy đủ.

### 7.5 Output của Assistant

Khi chạy dialogue script, output được ghi vào `nlp/app/output/` và mirror sang thư mục `python/output/` ở cấp package:

| File | Nội dung |
|---|---|
| `dialogue.txt` | Trace từng lượt: câu gốc, state, resolved query, retrieved chunks, answer/status, sources và state sau lượt |
| `answer.txt` | Câu trả lời cùng model/provider và citation |
| `assistant_run_summary.txt` | Thống kê lần chạy Assistant và RAG evidence đi kèm |

Chạy `--question` chỉ in kết quả ra terminal; không đi qua nhánh ghi batch outputs.

## 8. HTTP API

Khởi động server từ thư mục `python/`:

```bash
python -m nlp.app.api --host 127.0.0.1 --port 8000
```

Tùy chọn server: `--data-dir`, `--host`, `--port`, `--top-k`, `--min-score`, `--lexical-weight`, `--min-lexical-score`.

Các endpoint:

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/health` | Kiểm tra service và thông số retriever |
| `POST` | `/retrieve` | Trả chunks, scores và retrieval status |
| `POST` | `/answer` | Chạy Assistant và trả lời theo context/citation |
| `POST` | `/reset` | Reset DialogueState phía server |

Ví dụ:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Tuần 3 học gì?"}'

curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"Chương 4 nói về gì?"}'

curl -X POST http://127.0.0.1:8000/reset \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Endpoint `/answer` chấp nhận `question` (hoặc `query`) và có thể nhận `dialogue_state`/ `reset`. Endpoint `/retrieve` chấp nhận `query` (hoặc `question`); có thể bật `use_dialogue` hoặc gửi `dialogue_state`.

## 9. Sinh dataset có nhãn cho evaluation

### 9.1 Tạo lại samples

```bash
python -m nlp.app.build_sample_dataset
```

Mặc định đọc grammar `nlp/app/output/grammar.txt` và course data directory; ghi:

- `nlp/app/output/samples.txt`: mỗi câu một dòng.
- `nlp/app/output/samples_labeled.jsonl`: một JSON object mỗi dòng.

Tùy chỉnh:

```bash
python -m nlp.app.build_sample_dataset \
  --grammar nlp/app/output/grammar.txt \
  --data-dir python/data \
  --output-jsonl nlp/app/output/samples_labeled.jsonl \
  --output-text nlp/app/output/samples.txt \
  --max-sentences 10000
```

### 9.2 Schema và gán nhãn

Mỗi record gồm các trường chính:

| Field | Ý nghĩa |
|---|---|
| `sample_id` | ID của câu mẫu |
| `query` | Câu hỏi |
| `parse_status` | `PARSED` hoặc `UNPARSED` |
| `parse_tree` | Cây cú pháp chuẩn hóa nếu parse được |
| `intent` | Intent lấy từ nhánh gốc `Q_*` |
| `entities` | Danh sách entity IDs |
| `expected_chunk_ids` | Chunk IDs kỳ vọng trong Knowledge Base |
| `expected_answerability` | `EXPECTED_CONTEXT`, `INSUFFICIENT_CONTEXT`, `NO_RELEVANT_CONTEXT` hoặc `REVIEW` |
| `label_status` | Trạng thái rà soát nhãn |

Builder sinh câu từ grammar, parse lại, lấy intent từ nhánh `Q_*`, chuẩn hóa entity IDs và đối chiếu với chunk metadata. Các nhãn tự động được đánh dấu `AUTO_LABEL_REQUIRES_REVIEW`; **không được coi là gold label**.

Có nguy cơ rò rỉ đánh giá nếu nhãn intent/entity được tạo bằng cùng logic Semantic Frame mà Evaluation dùng để chấm. Tập grammar-generated phù hợp để kiểm tra độ phủ, parseability và regression; không đủ độc lập để tuyên bố semantic accuracy trên dữ liệu thực tế. Tương tự, expected chunks được suy ra bằng heuristic từ metadata nên cần audit trước khi dùng làm retrieval ground truth.

### 9.3 Tạo hàng đợi kiểm tra nhãn thay vì đọc toàn bộ 10.000 câu

Tạo một hàng đợi audit có chọn lọc, phân tầng theo intent, answerability, hình dạng parse tree, entity set và số expected chunks:

```bash
python -m nlp.app.label_audit \
  --input nlp/app/output/samples_labeled.jsonl \
  --output nlp/app/output/label_audit_queue.jsonl \
  --per-stratum 2 \
  --max-samples 300
```

Hàng đợi có các trường `auto_label`, `risk_flags` và `review` riêng. Người review điền `correct_intent`, `correct_entities`, `correct_expected_chunk_ids`, `correct_answerability`; nếu đánh giá parser thì điền thêm `correct_parse_status` và `correct_parse_tree`. Sau đó đổi `review_status` thành `REVIEWED` hoặc `REJECTED`. Không ghi đè nhãn tự động.

Sau khi review, xuất riêng tập gold chỉ gồm hàng đã duyệt:

```bash
python -m nlp.app.label_audit \
  --input nlp/app/output/samples_labeled.jsonl \
  --review-queue nlp/app/output/label_audit_queue.jsonl \
  --apply-reviewed \
  --reviewed-output nlp/app/output/samples_reviewed.jsonl
```

### 9.4 Chạy lại pipeline gán nhãn theo dữ liệu nguồn

Khi thay đổi dữ liệu trong `python/data/`, chạy:

```bash
python -m nlp.app.run_source_label_pipeline
```

Nếu đồng thời thay đổi grammar hoặc cần sinh lại toàn bộ sample:

```bash
python -m nlp.app.run_source_label_pipeline --rebuild-samples
```

Script này tạo lại hàng đợi audit, tự gán nhãn từ chunk hiện có, rồi xuất:

- `nlp/app/output/label_audit_queue.jsonl`
- `nlp/app/output/samples_reviewed.jsonl`

Các dòng reviewed có `label_provenance = "data_source_review"` để phân biệt với nhãn con người.

Chạy evaluation trên tập đã duyệt:

```bash
python -m nlp.app.evaluation --samples nlp/app/output/samples_reviewed.jsonl
python -m nlp.app.rag_evaluation --samples nlp/app/output/samples_reviewed.jsonl
```

Lệnh export từ chối hàng `REVIEWED` còn thiếu trường nhãn bắt buộc và không đưa hàng `PENDING`/`REJECTED` vào gold set.

Lấy 2 mẫu mỗi tầng là bước audit ban đầu, không phải chứng nhận tất cả câu cùng tầng đều đúng. Khi phát hiện một lỗi theo template/production, sửa rule hoặc grammar rồi sinh lại dataset và audit lại. Dùng thêm một tập câu hỏi viết tay, do người gán nhãn độc lập, làm held-out gold set cho metric chính thức. Nên đo độ đồng thuận giữa hai reviewer trên một phần mẫu nếu có đủ nhân lực.

## 10. Evaluation tổng thể và RAG evaluation

### 10.1 Lệnh evaluation chuẩn

Đây là lệnh chuẩn tạo báo cáo tổng hợp `evaluation.txt`:

```bash
python -m nlp.app.evaluation
```

Mặc định đọc `nlp/app/output/samples_labeled.jsonl` và ghi vào `nlp/app/output/`:

- `evaluation.txt`: báo cáo dễ đọc.
- `evaluation_report.json`: metrics chi tiết, confusion counts, per-sample failures, dataset SHA-256, runtime và cấu hình.

Chạy đầy đủ với data directory chỉ định:

```bash
python -m nlp.app.evaluation \
  --samples nlp/app/output/samples_labeled.jsonl \
  --data-dir python/data \
  --top-k 3 \
  --min-score 0.08
```

Chạy nhanh hoặc bỏ qua một số stage:

```bash
# Smoke run trên tối đa 100 mẫu (lấy mẫu đều theo thứ tự dataset)
python -m nlp.app.evaluation --max-samples 100

# Bỏ qua end-to-end và RAG evidence
python -m nlp.app.evaluation --skip-e2e

# Chỉ Parser + Semantic
python -m nlp.app.evaluation --skip-retrieval --skip-e2e
```

Các option: `--samples`, `--data-dir`, `--grammar`, `--output-dir`, `--max-samples`, `--top-k`, `--min-score`, `--skip-retrieval`, `--skip-e2e`.

### 10.2 Những gì được đo

| Stage | Metrics / kiểm tra |
|---|---|
| Parser / CFG | Parse-status accuracy, exact parse-tree match, confusion counts và failures |
| Semantic | Intent accuracy, entity micro-precision, micro-recall, micro-F1; missing/unexpected entities |
| Retrieval | Hit@k so với expected chunks, UNKNOWN rejection và hành vi với insufficient-context samples |
| End-to-End | Expected answerability/status confusion, retrieval hit rate trên mẫu có context, refusal accuracy, citation validity, NEEDS_REVIEW và over-answering |
| RAG evidence | Grounded examples, no-info/out-of-domain examples và scripted hallucination probes qua Answer Guard |

Retrieval trong aggregate evaluation chạy query-only, không đưa gold intent/entity labels vào retriever. End-to-end aggregate CLI dùng Mock LLM để tái lập và không phát sinh API cost; kết quả này không phải điểm chất lượng của LLM thật.

### 10.3 Giữ RAG evaluation độc lập

`rag_evaluation.py` vẫn là evaluator RAG riêng, không bị thay thế bởi `evaluation.py`:

```bash
python -m nlp.app.rag_evaluation
```

Mặc định tạo `nlp/app/output/rag_evaluation.txt`. Có thể chỉ định:

```bash
python -m nlp.app.rag_evaluation \
  --data-dir python/data \
  --provider mock \
  --max-samples 100
```

RAG report gồm labelled-set metrics, grounded/no-info/out-of-domain examples và scripted hallucination probes. Scripted model là test double được đưa qua pipeline thật để kiểm tra Answer Guard; **không phải LLM thật và không được báo cáo như kết quả model thật**. Khi chọn provider thật, mặc định evaluator giới hạn 50 labelled samples để hạn chế chi phí, trừ khi đặt `--max-samples`.

### 10.4 Metric interpretation và giới hạn

- Parse accuracy đo việc parser chấp nhận/từ chối đúng theo nhãn `parse_status`.
- Exact tree match đo độ trùng khớp cấu trúc cây với baseline trong JSONL.
- Intent accuracy là exact match.
- Entity precision/recall/F1 được micro-aggregate trên entity IDs.
- Hit@k là có ít nhất một expected chunk xuất hiện trong top-k.
- Refusal accuracy cần đọc cùng expected answerability; các mẫu `REVIEW` không nên được coi là nhãn chuẩn trước khi rà.
- Citation validity chỉ tính trên câu trả lời có status `ANSWERED`; nó không chứng minh entailment ngữ nghĩa đầy đủ.

Các cảnh báo bắt buộc khi phân tích kết quả:

1. `AUTO_LABEL_REQUIRES_REVIEW` phải được rà soát trước khi dùng số liệu làm kết luận học thuật.
2. Nếu thiếu sample data directory đầy đủ, Assistant có thể dùng trace fallback Knowledge Base; retrieval metrics lúc đó không đại diện cho Knowledge Base thật.
3. Grammar-generated samples đo tính nhất quán nội bộ, không chứng minh độ bao phủ hoặc độ tự nhiên với người dùng thật.
4. Mock LLM không đo chất lượng sinh ngôn ngữ của model thật.
5. Scripted hallucination probes kiểm tra guard theo các kịch bản cố định, không phải phép đo tần suất hallucination ngoài thực tế.

### 10.5 Đánh giá câu trả lời do LLM thật sinh ra

\`evaluation.py\` và \`rag_evaluation.py\` vẫn giữ vai trò hiện tại: parser/semantic/retrieval aggregate và RAG guard evidence. Để đo riêng chất lượng câu trả lời được sinh bởi model thật, dùng \`answer_evaluation.py\`:

\`\`\`bash
# Smoke test offline; kết quả này chỉ kiểm tra wiring/format, không phải điểm LLM thật
python -m nlp.app.answer_evaluation \\
  --samples nlp/app/input/test_labeled.jsonl \\
  --provider mock

# Chạy generator model thật, giới hạn số câu để kiểm soát chi phí
export OPENAI_API_KEY="..."
python -m nlp.app.answer_evaluation \\
  --samples nlp/app/input/answer_eval.jsonl \\
  --provider openai \\
  --model gpt-4o-mini \\
  --max-samples 30

# Bật Judge riêng; nên chọn model khác generator
export EVAL_JUDGE_MODEL="gpt-4o"
python -m nlp.app.answer_evaluation \\
  --samples nlp/app/input/answer_eval.jsonl \\
  --provider openai \\
  --model gpt-4o-mini \\
  --judge \\
  --judge-model "$EVAL_JUDGE_MODEL" \\
  --max-samples 30
\`\`\`

Schema JSONL tối thiểu cho answer evaluation:

\`\`\`json
{
  "sample_id": "ANS001",
  "query": "Chương 3 trình bày nội dung gì?",
  "expected_answerability": "EXPECTED_CONTEXT",
  "expected_chunk_ids": ["topics-chapter-3-grammar-and-parsing-2"],
  "label_status": "REVIEWED",
  "label_provenance": "human_review",
  "reference_answer": "Đáp án đã được đối chiếu với tài liệu.",
  "reference_points": ["Ý bắt buộc thứ nhất", "Ý bắt buộc thứ hai"],
  "reference_status": "HUMAN_REVIEWED"
}
\`\`\`

Quy tắc đánh giá:
- Headline answerability/retrieval metrics chỉ dùng dòng có \`label_status=REVIEWED\` và \`label_provenance=human_review\`. Các nhãn tự động, \`GOLD_REVIEWED\` hoặc \`data_source_review\` không được tự động coi là gold.
- Reference answer/points chỉ được gửi cho Judge khi \`reference_status=HUMAN_REVIEWED\`. Không dùng output do generator sinh ra làm đáp án chuẩn.
- Judge chấm correctness, faithfulness, relevancy, completeness (1–5) và hallucination flag. Đây là tín hiệu phụ trợ, không phải ground truth; cần đối chiếu với người đánh giá.
- Citation validity chỉ xác nhận citation thuộc context đã gửi, không chứng minh nội dung câu trả lời được context hỗ trợ về mặt ngữ nghĩa.
- Report ghi model ID thực tế, fallback, latency và chế độ Knowledge Base. Nếu dùng trace-fallback KB hoặc model fallback/mock, không diễn giải đó là kết quả của LLM thật trên dữ liệu đầy đủ.

Đầu ra: \`nlp/app/output/answer_evaluation/answer_evaluation.md\` và \`answer_evaluation.json\`.

## 11. Cài đặt và kiểm thử

Từ thư mục `python/`:

```bash
pip install -r requirements.txt
pip install -e .
```

Chạy test evaluation được thêm cho module aggregate:

```bash
python -m unittest discover -s test -p "test_evaluation*.py" -v
```

Có thể chạy toàn bộ unittest suite:

```bash
python -m unittest discover -s test -v
```

## 12. Các lệnh thường dùng — quick reference

Tất cả lệnh bên dưới chạy từ thư mục `python/`:

| Mục tiêu | Lệnh |
|---|---|
| Sinh câu | `python -m nlp.app.utils` |
| Parse câu | `python -m nlp.app.run_parser` |
| Sinh labelled dataset | `python -m nlp.app.build_sample_dataset` |
| Chạy một câu hỏi | `python -m nlp.app.run_assistant --question "Chương 4 nói về gì?"` |
| Chạy hội thoại | `python -m nlp.app.run_assistant --interactive` |
| Chạy HTTP API | `python -m nlp.app.api` |
| Retrieval độc lập | `python -m nlp.app.run_retrieval --labeled-jsonl nlp/app/output/samples_labeled.jsonl` |
| Evaluation tổng thể | `python -m nlp.app.evaluation` |
| RAG evaluation riêng | `python -m nlp.app.rag_evaluation` |
| Test evaluation | `python -m unittest discover -s test -p "test_evaluation*.py" -v` |

---

**Phạm vi thiết kế:** pipeline single-intent và multi-intent. Multi-intent chỉ được kích hoạt khi các clause được xác thực độc lập; nếu không, hệ thống giữ nguyên luồng single-intent. Khi code, grammar, CLI hoặc schema thay đổi, cập nhật README này để giữ tài liệu package đồng bộ với implementation.
