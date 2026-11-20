<<<<<<< Updated upstream
# BÀI TẬP LỚN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (CO3085)
## Trợ lý hỗ trợ học tập thông minh – Course Assistant

---

## 1. Thành viên nhóm
* **Môn học:** Xử lý ngôn ngữ tự nhiên (CO3085) – Học kỳ HK261
* **Khoa:** Khoa Khoa học và Kỹ thuật Máy tính – Trường Đại học Bách khoa – ĐHQG TP.HCM
* **Danh sách thành viên:**
  - Sinh viên 1: `Nguyễn Văn Bảo` – MSSV: `2433037`
  - Sinh viên 2: `Phạm Xuân Bắc` – MSSV: `2310280`
  - Sinh viên 3: `[Họ và tên]` – MSSV: `[MSSV3]`
  - Sinh viên 4: `[Họ và tên]` – MSSV: `[MSSV4]`

---

## 2. Hướng dẫn cài đặt

### 2.1 Yêu cầu hệ thống
* **Python:** `>= 3.8` (khuyến nghị Python `3.10+`, đã kiểm thử ổn định trên Python `3.12`)
* **Hệ điều hành:** Windows / Linux / macOS
* **Docker:** Phục vụ đóng gói môi trường chuẩn và chấm bài tự động

### 2.2 Cài đặt trực tiếp bằng `pip`
```bash
cd python
pip install -r requirements.txt
pip install -e .
```

### 2.3 Cài đặt và chạy qua Docker
```bash
# Build Docker image từ thư mục gốc của repository
docker build -t nlp-course-assistant python/

# Chạy container với mount thư mục input và output
docker run --rm -v "$(pwd)/output:/nlp/output" -v "$(pwd)/input:/nlp/input" nlp-course-assistant
```

---

## 3. Hướng dẫn chạy chương trình

Tất cả các lệnh dưới đây được thực thi từ thư mục `python/`:

```bash
# 1. Sinh câu mẫu hợp lệ từ văn phạm (Phần I - 3.2, tối đa 10.000 câu) -> nlp/app/output/samples.txt
python -m nlp.app.utils

# 2. Phân tích cú pháp tập câu kiểm thử (Phần I - 3.3) -> nlp/app/output/parse-results.txt
python -m nlp.app.run_parser \
  --input nlp/app/input/sentences.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt

# 3. Truy xuất Knowledge Base độc lập (Phần II - 4.2) -> nlp/app/output/retrieval.txt
python -m nlp.app.run_retrieval

# 4. Chạy toàn bộ pipeline Course Assistant (Phần II LLM/RAG, Phần III Hội thoại, Phần IV Đánh giá)
#    Tự động xuất dialogue.txt, answer.txt, evaluation.txt vào nlp/app/output/ và ../output/
python -m nlp.app.run_assistant --data-dir data

# 5. Chạy hỏi đáp 1 câu đơn lẻ hoặc chế độ hội thoại tương tác nhiều lượt
python -m nlp.app.run_assistant --data-dir data --question "Chương 6 nói về gì?" --json
python -m nlp.app.run_assistant --data-dir data --interactive

# 6. Khởi động HTTP API (/health, /retrieve, /answer, /reset)
python -m nlp.app.api --host 127.0.0.1 --port 8000

# 7. Chạy bộ kiểm thử tự động (46 test cases)
python -m pytest
```

---

## 4. Mô tả kiến trúc hệ thống

### 4.1 Sơ đồ kiến trúc tổng thể (End-to-End Pipeline)
Hệ thống được thiết kế theo kiến trúc phân tầng mô-đun (*modular layered pipeline*), giữ nguyên vai trò nền tảng của văn phạm và phân tích cú pháp ở Phần I, đồng thời tích hợp mượt mà với cơ chế truy xuất lai (*Hybrid Retrieval*), kiểm soát căn cứ (*Answer Guard*) của nhánh **LLM/RAG (Phần II)** và quản lý **ngữ cảnh hội thoại cục bộ (Phần III)**:

```text
====================================================================================================
                        KIẾN TRÚC PIPELINE HỆ THỐNG COURSE ASSISTANT
====================================================================================================

  [ Câu hỏi người dùng (Natural Language Input) ] + [ DialogueState (Ngữ cảnh lượt trước) ]
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 1: XỬ LÝ NGỮ CẢNH HỘI THOẠI CỤC BỘ (Phần III - nlp/app/dialogue.py)                 │
  │ - Phát hiện từ chỉ xuất / đại từ: "nó", "chương này", "phần này", "còn chương sau?"...   │
  │ - Nếu thiếu tiền từ (sau RESET) hoặc tiền từ mơ hồ (nhiều chương) -> NEED_CLARIFICATION  │
  │ - Nếu giải tham chiếu thành công -> Sinh câu hỏi đầy đủ: resolved_query                  │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ resolved_query
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 2: PHÂN TÍCH CÚ PHÁP CFG (Phần I - nlp/app/parser.py & output/grammar.txt)          │
  │ - Chuẩn hóa Unicode NFC, khớp cụm từ đa từ (multi-word terminals) không phân biệt hoa/thường│
  │ - Giải thuật Earley Chart Parser (Predictor, Scanner, Completer)                         │
  │ - Đầu ra: Cây cú pháp ParseNode dạng ngoặc (S (Q_... ...)) hoặc () nếu ngoài văn phạm   │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ ParseNode / None
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 3: DIỄN GIẢI NGỮ NGHĨA & INTENT/ENTITY (Phần II - semantic.py & semantic_frame.py)  │
  │ - Duyệt cây cú pháp theo nguyên lý thành phần (Compositionality)                         │
  │ - Ánh xạ nút gốc Q_* -> Intent (GET_COURSE_INFO, GET_TOPIC, GET_SCHEDULE, GET_LO, ...)   │
  │ - Chuẩn hóa thực thể qua từ điển đồng nghĩa data/entities.txt -> entity_ids & slots      │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ SemanticFrame {intent, entities, entity_ids, parse_status}
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 4: TRUY XUẤT TRI THỨC LAI (Phần II - knowledge_base.py & retriever.py)              │
  │ - Phân đoạn 7 file dữ liệu gốc (data/*.txt) thành các chunk gắn nhãn nguồn & entity_ids  │
  │ - Mở rộng truy vấn (Query Expansion) bằng alias + token thực thể + từ khóa intent        │
  │ - Tính điểm: CombinedScore = 0.75 * Cosine_TFIDF + 0.25 * SemanticOverlap                │
  │ - Chọn top_k = 3 chunk có điểm >= min_score (0.08) -> FOUND hoặc NO_RELEVANT_CONTEXT     │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ Retrieved Chunks + Scores
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 5: TIỀN KIỂM TRA & XÂY DỰNG PROMPT (answer_guard.py & prompt_builder.py)            │
  │ - Pre-LLM Guard (check_context): Chặn câu UNKNOWN hoặc không có từ vựng chung với chunk  │
  │ - Kiểm tra thuộc tính thiếu trong KB (ngày deadline chính thức, phòng học)               │
  │ - Đóng gói System Rules + Semantic Hint + Context Blocks có nhãn [Nguồn: file / section] │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ Grounded Prompt
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 6: SINH CÂU TRẢ LỜI & HẬU KIỂM TRA (llm_client.py & answer_guard.py)                │
  │ - Gọi MockLLMClient (offline) hoặc OpenAICompatibleLLMClient (online API)                │
  │ - Post-LLM Guard (check_answer):                                                         │
  │   (1) Bắt sentinel INSUFFICIENT_CONTEXT                                                  │
  │   (2) Xác thực mọi trích dẫn [Nguồn: ...] phải nằm trong các chunk đã gửi                │
  │   (3) Kiểm tra mọi số, ngày, giờ, email, URL phải tồn tại trong văn bản chunk            │
  │ - Cập nhật DialogueState cho lượt hội thoại tiếp theo                                    │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  [ Kết quả đầu ra: output/parse-results.txt, retrieval.txt, answer.txt, dialogue.txt, evaluation.txt ]
```

---

## 5. Mô tả Grammar

### 5.1 Cơ sở lý thuyết: Văn phạm phi ngữ cảnh (CFG) và Văn phạm ngữ nghĩa (Semantic Grammar)
1. **Văn phạm phi ngữ cảnh (CFG):**
   Một văn phạm phi ngữ cảnh được định nghĩa bởi bộ 4 thành phần $G = (V, \Sigma, R, S)$, trong đó:
   * $V$: Tập hữu hạn các ký hiệu phi kết thúc (*Non-terminals*, viết trong cặp dấu `<...>`).
   * $\Sigma$: Tập hữu hạn các ký hiệu kết thúc (*Terminals*, các từ hoặc cụm từ tiếng Việt/Anh đặt trong dấu `"..."`), rời nhau với $V$.
   * $R$: Tập các luật sản xuất (*Production rules*) có dạng $A \rightarrow \alpha$, với $A \in V$ và $\alpha \in (V \cup \Sigma)^*$.
   * $S \in V$: Ký hiệu bắt đầu đại diện cho một câu hoàn chỉnh.
2. **Văn phạm ngữ nghĩa (Semantic Grammar):**
    (*Other Strategies for Semantic Interpretation*), đối với các hệ thống hỏi đáp trong miền ứng dụng hẹp (*restricted-domain QA*), thay vì chỉ sử dụng các phạm trù thuần cú pháp (`NP`, `VP`, `PP`), ta thiết kế các ký hiệu phi kết thúc mang **ý nghĩa miền bài toán** (như `<Q_COURSE_INFO>`, `<Q_SCHEDULE>`, `<CHAPTER_REF>`, `<TOPIC_REF>`, `<LO_REF>`). Cách tiếp cận này giúp:
   * Giảm đáng kể độ nhập nhằng cấu trúc (*structural ambiguity*);
   * Ràng buộc trực tiếp các giới hạn lựa chọn (*selectional restrictions*): ví dụ động từ/cụm vị ngữ hỏi thời lượng thi (`<EXAM_TAIL> ::= "bao nhiêu phút?"`) chỉ kết hợp với `<EXAM_REF>` (`midterm`, `final`), còn cụm vị ngữ hỏi tín chỉ chỉ kết hợp với `<COURSE_REF>`;
   * Cho phép suy ra trực tiếp **Logical Form / Semantic Frame** từ chính cấu trúc cây cú pháp mà không cần tầng biến đổi trung gian phức tạp.

### 5.2 Thiết kế chi tiết tập luật trong `output/grammar.txt`
File văn phạm `output/grammar.txt` (được sinh và quản lý bởi `course_grammar.py` và `grammar_file.py`) định nghĩa ký hiệu gốc `<S>` phân nhánh thành **9 phạm trù câu hỏi cấp cao**:

```bnf
<S> ::= <Q_COURSE_INFO>
      | <Q_SCHEDULE>
      | <Q_TOPIC>
      | <Q_ASSIGNMENT>
      | <Q_DEADLINE>
      | <Q_RESOURCE>
      | <Q_RULE>
      | <Q_LO>
      | <Q_UNKNOWN>
```

Chi tiết từng nhóm luật trong văn phạm:
1. **Nhóm hỏi thông tin chung môn học (`<Q_COURSE_INFO>`):**
   * Hỗ trợ hỏi số tín chỉ, mã môn học, mô tả môn học, cấu trúc các phần và thời lượng bài thi giữa kỳ/cuối kỳ, có hoặc không có tiền tố lịch sự (`<POLITE_PREFIX>`):
   ```bnf
   <Q_COURSE_INFO> ::= <COURSE_REF> <COURSE_INFO_TAIL>
                     | <POLITE_PREFIX> <COURSE_REF> <COURSE_INFO_TAIL>
                     | <EXAM_REF> <EXAM_TAIL>
                     | <POLITE_PREFIX> <EXAM_REF> <EXAM_TAIL>
   <COURSE_REF>    ::= "Môn NLP" | "môn học" | "CO3085" | "C03085" | "môn CO3085" | "môn C03085" | "Xử lý Ngôn ngữ Tự nhiên"
   <EXAM_REF>      ::= "Midterm" | "midterm" | "Final" | "final" | "thi giữa kỳ" | "thi cuối kỳ"
   <EXAM_TAIL>     ::= "bao nhiêu phút?" | "kéo dài bao nhiêu phút?" | "thi bao lâu?"
   ```
2. **Nhóm hỏi lịch học và kế hoạch giảng dạy (`<Q_SCHEDULE>`):**
   * Hỗ trợ hỏi nội dung theo tuần (`<WEEK_REF>`), hỏi tuần học của một chương (`<CHAPTER_REF>`) hoặc một chủ đề (`<TOPIC_REF>`), và hỏi lịch học chung:
   ```bnf
   <Q_SCHEDULE> ::= <WEEK_REF> <SCHEDULE_WEEK_TAIL>
                  | <CHAPTER_REF> <SCHEDULE_CHAPTER_TAIL>
                  | <SCHEDULE_TOPIC_REF> <SCHEDULE_TOPIC_TAIL>
                  | "Khi nào học" <TOPIC_REF> "?"
                  | <DEICTIC_CHAPTER> <SCHEDULE_CHAPTER_TAIL>
                  | <DEICTIC_CONTENT> <SCHEDULE_CHAPTER_TAIL>
                  | <SCHEDULE_WHEN>
   ```
3. **Nhóm hỏi nội dung và chủ đề bài học (`<Q_TOPIC>`):**
   * Hỗ trợ hỏi nội dung một chương (`Chương 6 nói về gì?`), tìm chương chứa chủ đề (`Chương nào nói về PCFG?`, `Chương nào đề cập đến gán nhãn từ loại?`), kiểm tra mối liên hệ giữa chương và chủ đề (`Chương 6 có liên quan đến gán nhãn từ loại không?`), so sánh hai chương (`Chương 2 và Chương 3 khác nhau như thế nào?`) và yêu cầu giải thích/giới thiệu chủ đề (`giới thiệu về mạng ngữ nghĩa`):
   ```bnf
   <Q_TOPIC> ::= <CHAPTER_REF> <TOPIC_CHAPTER_TAIL>
               | "Chương nào nói về" <TOPIC_REF> "?"
               | "Chương nào đề cập đến" <TOPIC_REF> "?"
               | "Chương nào học về" <TOPIC_REF> "?"
               | <CHAPTER_REF> <TOPIC_RELATED_TAIL>
               | <DEICTIC_CHAPTER> <TOPIC_RELATED_TAIL>
               | <CHAPTER_REF> "và" <CHAPTER_REF> <TOPIC_COMPARE_TAIL>
               | "giải thích" <TOPIC_REF> | "giới thiệu về" <TOPIC_REF>
   ```
4. **Nhóm hỏi thông tin bài tập lớn (`<Q_ASSIGNMENT>`) và hạn nộp (`<Q_DEADLINE>`):**
   * Bao phủ câu hỏi về yêu cầu chung của bài tập lớn, nội dung từng phần (`Phần I` – `Phần IV`), câu hỏi Yes/No về thành phần bài tập (`Bài tập lớn có phần hội thoại không?`), trách nhiệm nộp bài của sinh viên (`Tôi cần hoàn thành những bài tập nào?`, `Sinh viên phải nộp những gì?`) và thời hạn nộp bài (`Deadline của bài tập lớn là khi nào?`, `Hạn nộp bài tập lớn là khi nào?`).
5. **Nhóm hỏi tài liệu tham khảo (`<Q_RESOURCE>`):**
   * Hỗ trợ tìm tài liệu theo chủ đề (`Tài liệu nào nói về PCFG?`, `Tài liệu nào liên quan đến đồng tham chiếu?`), theo phạm vi chương/tuần/phần (`Tài liệu tham khảo cho Chương 6 là gì?`, `Tài liệu cho Phần I ở đâu?`), theo nhánh chuyên ngành (`statistical NLP`, `Python NLP`, `machine translation`) và câu mệnh lệnh (`liệt kê tài liệu tham khảo`, `tìm tài liệu về semantic grammar`).
6. **Nhóm hỏi quy định môn học (`<Q_RULE>`) và chuẩn đầu ra (`<Q_LO>`):**
   * `<Q_RULE>` kiểm tra quy định sử dụng công cụ AI, mở rộng LLM/RAG, thư viện ngoài và trích dẫn nguồn.
   * `<Q_LO>` truy vấn nội dung và năng lực yêu cầu của các chuẩn đầu ra `L.O.2.5`, `L.O.3.3`, `L.O.4.2`.
7. **Nhóm câu hỏi ngoài miền mẫu (`<Q_UNKNOWN>`):**
   * Nhận diện trực tiếp các mẫu câu ngoài phạm vi môn học để kiểm thử tầng lọc ngữ nghĩa.

### 5.3 Giải thuật sinh câu từ văn phạm (`sentence_generator.py`, `utils.py`)
* **Nguyên lý:** Trong `SentenceGenerator.generate()`, quá trình sinh câu được mô hình hóa như bài toán duyệt đồ thị các dẫn xuất trái nhất (*leftmost derivations*) theo **chiều rộng (Breadth-First Search – BFS)** sử dụng cấu trúc hàng đợi `collections.deque`:
  1. Khởi tạo hàng đợi `pending = deque([([grammar.start_symbol], 0)])`, danh sách kết quả `sentences = []` và tập khử trùng lặp `seen = set()`.
  2. Ở mỗi bước lặp (khi `pending` còn phần tử và `len(sentences) < max_sentences`):
     * Lấy ra trạng thái `(symbols, expansion_count)` ở đầu hàng đợi.
     * Tìm vị trí `index` của ký hiệu phi kết thúc đầu tiên từ trái sang phải (`grammar.is_nonterminal(symbol)`).
     * **Trường hợp cơ sở (`index is None`):** Mọi ký hiệu trong `symbols` đều là ký hiệu kết thúc (*terminals*). Ghép các token bằng khoảng trắng, chuẩn hóa khoảng trắng trước dấu câu (`re.sub(r"\s+([,.;:!?])", r"\1", sentence)`), và nếu câu chưa xuất hiện trong `seen` thì thêm vào `sentences`.
     * **Bước mở rộng (`index is not None`):** Nếu `expansion_count < max_expansions` (mặc định `30` để chặn đệ quy vô hạn), duyệt qua mọi vế phải `production` của `symbols[index]` và đẩy dãy ký hiệu mới `symbols[:index] + list(production) + symbols[index + 1:]` kèm `expansion_count + 1` vào cuối hàng đợi `pending`.
* **Ưu điểm của BFS:** Đảm bảo sinh tất cả các câu ngắn, cấu trúc cơ bản và bao phủ đồng đều mọi nhánh `<Q_*>` trước khi mở rộng sang các biến thể dài có tiền tố lịch sự.
* **Kết quả thực thi:** Sinh ra **8.926 câu** hợp lệ, không trùng lặp (nằm trong giới hạn $\le 10.000$ câu của đề bài) và lưu tại `output/samples.txt`.

---

## 6. Mô tả Parser

### 6.1 Cơ sở lý thuyết: Chart Parsing và Giải thuật Earley
* Trong bài giảng Chương 3 (*Grammar and Parsing*), các giải thuật phân tích cú pháp quay lui đơn thuần (*Simple Top-Down / Bottom-Up Parser*) gặp nhược điểm lớn là phải phân tích đi phân tích lại cùng một cụm từ con khi một nhánh giả thuyết thất bại.
* **Chart Parsing** khắc phục triệt để nhược điểm này bằng cách sử dụng cấu trúc bảng **Chart** lưu các cung hoạt động (*active arcs / dotted rules*) và các thành phần đã hoàn tất (*completed constituents*).
* Hệ thống cài đặt **Earley Chart Parser** (`CourseParser` trong `nlp/app/parser.py`), hoạt động trực tiếp trên mức ký tự/cụm từ của câu đã chuẩn hóa mà không đòi hỏi phải chuyển văn phạm về dạng chuẩn Chomsky (CNF).

### 6.2 Cấu trúc dữ liệu và Giải thuật chi tiết trong `parser.py`
Cho câu đầu vào `sentence`, sau khi chuẩn hóa Unicode dựng sẵn `text = unicodedata.normalize("NFC", sentence.strip())` có độ dài $N = \text{len(text)}$:
* **Biểu diễn trạng thái Earley (Dotted Rule):** Mỗi trạng thái là một bộ 5 phần tử:
  $$\text{state} = (\text{lhs}, \text{prod}, \text{dot}, \text{origin}, \text{children})$$
  tương ứng với luật $\text{lhs} \rightarrow \alpha \bullet \beta$ bắt đầu tại vị trí ký tự $\text{origin}$, đã khớp được $\text{dot}$ ký hiệu trong $\text{prod}$ và tích lũy danh sách các nút con $\text{children}$.
* **Bảng `chart` và hàng đợi `agenda`:** Khởi tạo `chart = [set() for _ in range(N + 1)]` (dùng `set` để kiểm tra trùng lặp trạng thái trong $O(1)$) và `agenda = [[] for _ in range(N + 1)]` (duyệt tuần tự các trạng thái tại mỗi vị trí). Hàm `add(pos, ...)` tự động bỏ qua khoảng trắng (`_skip_spaces`) trước khi chèn trạng thái vào `chart[pos]` và `agenda[pos]`.
* **Khởi tạo:** Thêm luật giả khởi đầu `("__START__", (("N", "S"),), dot=0, origin=0, children=())` vào vị trí `pos = 0`.
* **Vòng lặp chính ($0 \le \text{pos} \le N$):** Duyệt từng trạng thái trong `agenda[pos]` và áp dụng 1 trong 3 phép toán:
  1. **Predictor (Dự đoán – khi ký hiệu sau dấu chấm là Non-terminal `("N", sym)`):**
     * Với mọi luật con `child_prod` của `sym`, thêm trạng thái `(sym, child_prod, 0, pos, ())` vào `chart[pos]`.
     * *Xử lý luật rỗng (Nullable `<EMPTY>`):* Nếu tại chính vị trí `pos` đã có trạng thái của `sym` hoàn tất tức thời (`d_orig == pos` và `d_dot == len(d_prod)`), tiến ngay dấu chấm của trạng thái hiện tại lên `dot + 1`.
  2. **Scanner (Quét từ vựng – khi ký hiệu sau dấu chấm là Terminal `("T", sym)`):**
     * Kiểm tra đoạn văn bản `text[pos : pos + len(sym)]` có khớp không phân biệt hoa thường (`casefold()`) với `sym` hay không.
     * Kiểm tra điều kiện biên từ (*word-boundary check*): vị trí kết thúc `m_end = pos + len(sym)` phải nằm ở cuối câu, hoặc theo sau bởi khoảng trắng, hoặc theo sau bởi dấu câu `?!,.;:` (ngăn lỗi khớp tiền tố như khớp `"môn"` vào giữa một từ dài).
     * Nếu khớp, thêm trạng thái mới tại vị trí `m_end` với `dot + 1` và gắn nút lá `ParseNode(sym)` vào `children`.
  3. **Completer (Hoàn tất – khi `dot == len(prod)`):**
     * Tạo nút cú pháp hoàn chỉnh `node = ParseNode(lhs, children)`.
     * Duyệt các trạng thái cha `(p_lhs, p_prod, p_dot, p_orig, p_ch)` trong `chart[origin]`; nếu trạng thái cha đang chờ đúng `("N", lhs)` tại `p_dot`, thêm trạng thái cha đã tiến dấu chấm `p_dot + 1` kèm `p_ch + (node,)` vào `chart[pos]`.
* **Điều kiện chấp nhận và Đầu ra:**
  * Sau khi duyệt tới `final_pos = _skip_spaces(text, N)`, nếu trong `agenda[final_pos]` tồn tại trạng thái `lhs == "__START__"`, `dot == 1`, `orig == 0`, trả về cây `ParseNode("S", ...)` và xuất chuỗi ngoặc đơn qua `node.to_bracketed()`.
  * Ngược lại, trả về `None` và hàm `parse_line()` xuất đúng ký hiệu `()` theo yêu cầu mục 3.3.
  * Hàm `parse()` được bọc bởi `@lru_cache(maxsize=50000)` giúp tăng tốc độ phân tích hàng loạt khi chạy đánh giá trên hàng nghìn mẫu.

### 6.3 Ví dụ kết quả phân tích cú pháp (`output/parse-results.txt`)
* **Câu hợp lệ (thuộc văn phạm):**
  * Đầu vào: `Cho hỏi CO3085 có bao nhiêu tín chỉ?`
  * Đầu ra: `(S (Q_COURSE_INFO (POLITE_PREFIX (cho hỏi)) (COURSE_REF (CO3085)) (COURSE_INFO_TAIL (có bao nhiêu tín chỉ?))))`
  * Đầu vào: `Chương 6 có liên quan đến gán nhãn từ loại không?`
  * Đầu ra: `(S (Q_TOPIC (CHAPTER_REF (chương) (CHAPTER_NUMBER (6))) (TOPIC_RELATED_TAIL (có liên quan đến) (TOPIC_REF (gán nhãn từ loại)) (không?))))`
* **Câu không hợp lệ / ngoài văn phạm:**
  * Đầu vào: `Thời tiết hôm nay thế nào?`
  * Đầu ra: `()`

---

## 7. Mô tả Semantic Representation

### 7.1 Cơ sở lý thuyết: Dạng luận lý (Logical Form) và Nguyên lý thành phần (Compositionality)
* **Logical Form (Chương 7):** Biểu diễn ngữ nghĩa độc lập ngữ cảnh (*context-independent meaning representation*) đóng vai trò cầu nối giữa phân tích cú pháp và cơ sở tri thức. Đối với các câu hỏi trong hệ thống hỏi đáp, dạng luận lý xác định rõ **tác tử truy vấn (Query Operator / Intent)** và **các đối số thực thể được ràng buộc (Bound Arguments / Thematic Slots)**.
* **Principle of Compositionality (Chương 8):** Nghĩa của một cấu trúc phức hợp được tổng hợp từ nghĩa của các thành phần con và quy tắc kết hợp cú pháp của chúng. Nhờ thiết kế Semantic Grammar ở Phần I, cấu trúc phân cấp của `ParseNode` cho phép tổng hợp khung ngữ nghĩa một cách tất định.

### 7.2 Cấu trúc `SemanticFrame` và Biểu diễn vị từ (`semantic.py`, `semantic_frame.py`)
Trong hệ thống, mỗi câu hỏi sau khi phân tích cú pháp được biểu diễn dưới hai dạng tương đương:
1. **Biểu diễn vị từ tường minh (Predicate-Argument Representation):**
   Có dạng `INTENT_PREDICATE(slot_1=ENTITY_1, slot_2=ENTITY_2, ...)`:
   * `GET_CREDIT(course=CO3085)` hoặc `GET_COURSE_INFO(exam=MIDTERM)`
   * `GET_SCHEDULE(week=WEEK_02)` hoặc `GET_SCHEDULE(chapter=CH06)` hoặc `GET_SCHEDULE(topic=DEPENDENCY_GRAMMAR)`
   * `GET_TOPIC(chapter=CH06)` hoặc `GET_TOPIC(chapter=CH06, topic=POS_TAGGING)` hoặc `GET_TOPIC(chapter_1=CH02, chapter_2=CH03)`
   * `GET_ASSIGNMENT(part=PART_II)` hoặc `GET_ASSIGNMENT(assignment=BTL01)`
   * `GET_DEADLINE(assignment=BTL01)`
   * `GET_RESOURCE(topic=PCFG)` hoặc `GET_RESOURCE(chapter=CH06)` hoặc `GET_RESOURCE(week=WEEK_08)`
   * `GET_RULE(rule=AI_USAGE)` hoặc `GET_RULE(rule=LLM_RAG)`
   * `GET_LO(lo=LO3.3)`
2. **Cấu trúc `SemanticFrame` (Dictionary chuẩn giao tiếp giữa các tầng):**
   ```json
   {
     "intent": "GET_TOPIC",
     "entities": {
       "chapter": "CH06",
       "topic": "POS_TAGGING"
     },
     "entity_ids": ["CH06", "POS_TAGGING"],
     "parse_status": "PARSED"
   }
   ```
   * Khi câu đầu vào không thuộc văn phạm (`parse_tree is None`), hàm `unparsed_frame()` trong `semantic.py` trả về khung trung lập `{"intent": None, "entities": {}, "entity_ids": [], "parse_status": "UNPARSED"}` (tuyệt đối không đoán mò intent sai lệch).

### 7.3 Khung ngữ nghĩa phía tài liệu (`chunk_frame`)
Không chỉ câu hỏi mà mỗi đoạn tài liệu (chunk) trong Knowledge Base cũng được gắn một khung ngữ nghĩa tại thời điểm lập chỉ mục qua hàm `chunk_frame(document)` trong `semantic_frame.py`:
```python
KIND_TO_INTENTS = {
    "course_info": {"GET_COURSE_INFO", "GET_LO"},
    "schedule":    {"GET_SCHEDULE"},
    "topics":      {"GET_TOPIC"},
    "assignments": {"GET_ASSIGNMENT", "GET_DEADLINE"},
    "resources":   {"GET_RESOURCE"},
    "regulations": {"GET_RULE"},
    "faq":         {"GET_COURSE_INFO", "GET_SCHEDULE", "GET_TOPIC", "GET_ASSIGNMENT",
                    "GET_DEADLINE", "GET_RESOURCE", "GET_RULE", "GET_LO"},
}
```
Việc biểu diễn cả câu hỏi lẫn tài liệu trong cùng một không gian `SemanticFrame` (`intents` + `entity_ids`) cho phép hệ thống thực hiện đối sánh ngữ nghĩa chính xác ở tầng Retriever.

---

## 8. Mô tả phương pháp xử lý Intent/Entity

### 8.1 Phương pháp nhận diện Intent (`semantic_frame.py`)
* **Ánh xạ từ nút cú pháp cấp câu (`TREE_INTENT_TO_INTENT`):**
  Hàm `frame_from_tree(tree, query, aliases)` tìm nút con trực tiếp của `S` có nhãn bắt đầu bằng `"Q_"` và tra bảng ánh xạ 1–1:
  - `Q_COURSE_INFO` $\rightarrow$ `GET_COURSE_INFO`
  - `Q_SCHEDULE` $\rightarrow$ `GET_SCHEDULE`
  - `Q_TOPIC` $\rightarrow$ `GET_TOPIC`
  - `Q_ASSIGNMENT` $\rightarrow$ `GET_ASSIGNMENT`
  - `Q_DEADLINE` $\rightarrow$ `GET_DEADLINE`
  - `Q_RESOURCE` $\rightarrow$ `GET_RESOURCE`
  - `Q_RULE` $\rightarrow$ `GET_RULE`
  - `Q_LO` $\rightarrow$ `GET_LO`
  - `Q_UNKNOWN` $\rightarrow$ `UNKNOWN`

### 8.2 Phương pháp trích xuất và chuẩn hóa Entity (`semantic_frame.py`, `semantic.py`, `knowledge_base.py`)
Quá trình trích xuất thực thể kết hợp **3 cơ chế bổ trợ lẫn nhau**:
1. **Duyệt nút thực thể trên cây cú pháp (`_first_node` & `_node_text`):**
   Duyệt theo thứ tự ưu tiên qua 13 loại nhãn nút thực thể trên cây cú pháp:
   * `COURSE_REF` $\rightarrow$ chuẩn hóa về `"CO3085"`.
   * `EXAM_REF` $\rightarrow$ kiểm tra bề mặt (`"midterm"` / `"giữa kỳ"` $\rightarrow$ `"MIDTERM"`, ngược lại $\rightarrow$ `"FINAL"`).
   * `WEEK_REF` $\rightarrow$ trích số nguyên bằng regex `\d+` và định dạng thành `"WEEK_{:02d}"` (`WEEK_01` .. `WEEK_15`).
   * `CHAPTER_REF` $\rightarrow$ trích số nguyên và định dạng thành `"CH{:02d}"` (`CH01` .. `CH12`).
   * `LO_REF` $\rightarrow$ trích chuỗi `\d+\.\d+` và định dạng thành `"LO2.5"`, `"LO3.3"`, `"LO4.2"`.
   * `ASSIGNMENT_REF`, `DEADLINE_REF` $\rightarrow$ chuẩn hóa về `"BTL01"`.
   * `RULE_OBJECT`, `RULE_REF` $\rightarrow$ phân biệt `"LLM_RAG"` và `"AI_USAGE"`.
   * `RESOURCE_TOPIC` $\rightarrow$ phân biệt `"PYTHON_NLP"`, `"STATISTICAL_NLP"`, `"MACHINE_TRANSLATION"`.
   * `TOPIC_REF`, `PART_REF`, `RESOURCE_REF` $\rightarrow$ tra cứu trong từ điển đồng nghĩa `aliases`.
2. **Đối sánh từ điển thực thể và đồng nghĩa (`load_entity_aliases` từ `data/entities.txt`):**
   * File `data/entities.txt` định nghĩa các cặp `alias = canonical_value` theo từng nhóm (`course`, `chapters`, `topics`, `learning_outcomes`, `assignments`, `policies`).
   * Hàm `load_entity_aliases()` chuẩn hóa các tên gọi tiếng Việt và tiếng Anh về cùng một `entity_id` (ví dụ: `"văn phạm phi ngữ cảnh"`, `"context-free grammar"`, `"CFG"` $\rightarrow$ `CFG`; `"gán nhãn từ loại"`, `"part-of-speech tagging"`, `"POS tagging"` $\rightarrow$ `POS_TAGGING`) và **sắp xếp giảm dần theo độ dài chuỗi** (`key=lambda item: len(item[0]), reverse=True`) để cụm từ cụ thể dài hơn luôn được ưu tiên khớp trước.
3. **Bắt thực thể số học theo mẫu cấu trúc (Structure-aware Regex Fallback):**
   * Quét bổ sung trên câu đã chuẩn hóa với biểu thức `\b(?:tuan|week)\s+(?:thu\s+)?0?(\d{1,2})\b` và `\b(?:chuong|chapter)\s+0?(\d{1,2})\b` để đảm bảo bắt đủ mọi chương/tuần ngay cả trong câu so sánh có nhiều chương (`"Chương 2 và Chương 3 khác nhau như thế nào?"` $\rightarrow$ `["CH02", "CH03"]`).
4. **Phân loại khe ngữ nghĩa (`entity_category` trong `semantic.py`):**
   * Gán từng `entity_id` vào đúng slot ngữ nghĩa (`chapter`, `week`, `part`, `assignment`, `course`, `lo`, `exam`, `rule`, `topic`) trong dictionary `entities` để phục vụ cập nhật `DialogueState`.

---

## 9. Mô tả Knowledge Base và cách truy vấn

### 9.1 Cơ sở lý thuyết: Biểu diễn tri thức và Truy xuất thông tin
* Trong bài giảng **Chương 1 (Information Retrieval & Information Extraction)** và **Chương 11 (Knowledge Representation and Reasoning – Procedural Semantics and Question Answering)**, một hệ thống hỏi đáp miền hẹp cần tổ chức cơ sở tri thức sao cho vừa bảo toàn ngữ cảnh văn bản tự nhiên, vừa gắn kết với các ký hiệu thực thể chuẩn hóa để thực hiện truy vấn chính xác.

### 9.2 Cấu trúc Knowledge Base và Phân đoạn dữ liệu (`knowledge_base.py`)
Knowledge Base được nạp từ 7 tập tin văn bản cộng 1 tập tin thực thể trong `python/data/` (nguồn dữ liệu chuẩn của môn học CO3085 do giảng viên cung cấp):

| file | Loại | Quy tắc phân đoạn | Nội dung chính |
|---|---|---|---|
| `course_info.txt` | `course_info` | Tách theo đoạn trống `\n\s*\n` | Mã môn `CO3085`, 3 tín chỉ, phân bổ giờ, mô tả 3 phần, chuẩn đầu ra `L.O.`. |
| `schedule.txt` | `schedule` | Tách theo tiêu đề `(?=^WEEK\s+\d+\b)` | Kế hoạch giảng dạy chi tiết từ `WEEK 01` đến `WEEK 15`, lịch thi Midterm/Final. |
| `topics.txt` | `topics` | Tách theo tiêu đề `(?=^CHAPTER\s+\d+\s*:)` | Nội dung 12 chương (`CHAPTER 1` – `CHAPTER 12`): `part`, `keywords`, `summary`, `subtopics`. |
| `assignments.txt` | `assignments` | Tách theo `(?=^(?:PART\s+[IVX]+\s*-|LLM/RAG EXTENSION))` | Yêu cầu Phần I, II, III, IV của bài tập lớn, các file output cần nộp, lưu ý deadline. |
| `regulations.txt` | `regulations` | Tách theo tên trường quy định (`attendance:`, `AI_usage:`, `LLM_RAG:`, ...) | Quy định điểm danh, trung thực học thuật, sử dụng AI, LLM/RAG, thư viện ngoài. |
| `resources.txt` | `resources` | Tách theo `(?=^(?:TEXTBOOK\s+\d+|REFERENCE\s+\d+|study_note\s*:))` | 3 giáo trình chính (Allen 1995; Jurafsky & Martin; Manning & Schütze) và 5 tài liệu tham khảo. |
| `faq.txt` | `faq` | Tách theo câu hỏi `(?=^Q\d+\s*:)` | Các cặp câu hỏi – trả lời thường gặp của môn học (`Q1` – `Q15`). |

Mỗi đoạn sau khi tách được đóng gói thành một dictionary **chunk**:
```python
{
    "chunk_id": "topics-chapter-6-ambiguity-resolution-statistical-methods-5",
    "text": "CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS ...",
    "source": "topics.txt",
    "section": "CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS",
    "entity_ids": ["CH06", "SYNTACTIC_PROCESSING", "POS_TAGGING", "PCFG"],
    "kind": "topics",
}
```

### 9.3 Cơ chế truy vấn lai (Hybrid Retrieval: TF-IDF + Semantic Frame trong `retriever.py`)
Khi nhận câu hỏi `question`, `Retriever.retrieve()` thực hiện quy trình 4 bước:
1. **Xây dựng chỉ mục TF-IDF có làm giàu thực thể (`__init__`):**
   * Mỗi chunk trong corpus được nối thêm chuỗi token đặc trưng `entityid<id>` (ví dụ `entityidch06`, `entityidpcfg`) lặp lại 2 lần nhằm tăng trọng số khớp thực thể chính xác.
   * Khởi tạo `TfidfVectorizer(preprocessor=normalize, lowercase=True, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)` với thang đo tần suất dưới tuyến tính $1 + \log(\text{tf})$.
2. **Mở rộng câu truy vấn (`expand_query`):**
   * Đối sánh các alias trong câu hỏi để chèn thêm từ khóa chuẩn tiếng Anh/Việt (`canonical`) và token `entityid<id>`.
   * Dựa trên `intent` của `SemanticFrame`, chèn thêm bộ từ khóa định hướng miền (`intent_terms`), ví dụ:
     - `GET_SCHEDULE` $\rightarrow$ `"schedule week weekly plan lesson topics"`
     - `GET_TOPIC` $\rightarrow$ `"chapter topic summary subtopics"`
     - `GET_RESOURCE` $\rightarrow$ `"resources textbook reference books"`
3. **Tính điểm kết hợp (Hybrid Scoring):**
   * **Điểm từ vựng ($\text{LexicalScore}$):** Độ tương đồng Cosine giữa vector TF-IDF của truy vấn mở rộng và ma trận chunk:
     $$\text{LexicalScore}(q, d) = \cos(\mathbf{v}_{\text{TF-IDF}}(q_{\text{expanded}}), \mathbf{v}_{\text{TF-IDF}}(d))$$
   * **Điểm ngữ nghĩa ($\text{SemanticScore}$ – hàm `_semantic_scores`):**
     Gọi $I_q$ là intent của câu hỏi, $E_q$ là tập thực thể của câu hỏi, $I_d$ và $E_d$ là tập intent và thực thể của chunk $d$:
     - Nếu $E_q = \emptyset$ (câu hỏi tổng quát không nêu thực thể cụ thể):
       $$\text{SemanticScore}(q, d) = \begin{cases} 1.0 & \text{nếu } I_q \in I_d \text{ và } \text{kind}(d) \neq \text{faq} \\ 0.1 & \text{nếu } I_q \in I_d \text{ và } \text{kind}(d) = \text{faq} \\ 0.0 & \text{ngược lại} \end{cases}$$
     - Nếu $E_q \neq \emptyset$ (câu hỏi có thực thể cụ thể):
       $$\text{SemanticScore}(q, d) = \text{IntentPrior}(q, d) + 0.8 \times \frac{|E_q \cap E_d|}{|E_q|}$$
       với $\text{IntentPrior}(q, d) = 0.2$ nếu $I_q \in I_d$ (hoặc $0.05$ nếu là chunk `faq`).
   * **Điểm tổng hợp ($\text{CombinedScore}$):** Với trọng số mặc định `lexical_weight = 0.75` và `semantic_weight = 0.25`:
     $$\text{CombinedScore}(q, d) = 0.75 \times \text{LexicalScore}(q, d) + 0.25 \times \text{SemanticScore}(q, d)$$
     *(Nếu câu hỏi `UNPARSED` không có intent hợp lệ, hệ thống tự động dùng $\text{CombinedScore}(q, d) = \text{LexicalScore}(q, d)$).*
4. **Lọc theo ngưỡng và Trả kết quả:**
   * Nếu `intent == "UNKNOWN"`, danh sách kết quả được xóa rỗng ngay lập tức (`selected = []`).
   * Ngược lại, sắp xếp giảm dần theo `CombinedScore` và lấy tối đa `top_k = 3` chunk thỏa mãn $\text{CombinedScore} \ge \text{min\_score} = 0.08$.
   * Trả về `retrieval_status = "FOUND"` nếu có ít nhất 1 chunk được chọn, hoặc `"NO_RELEVANT_CONTEXT"` nếu rỗng.

---

## 10. Mô tả lựa chọn LLM/RAG

Nhóm lựa chọn **Lựa chọn 2: Mở rộng bằng LLM/RAG** đồng thời hoàn thiện đầy đủ **Xử lý ngữ cảnh hội thoại cục bộ**. Đúng theo yêu cầu của đề bài, phần mở rộng LLM/RAG **không thay thế** các thành phần NLP cốt lõi mà sử dụng trực tiếp kết quả của Grammar, Earley Parser và Semantic Frame từ Phần I.

### 10.1 Mô hình sử dụng và Cấu hình (`llm_client.py`)
Hệ thống định nghĩa giao diện trừu tượng `LLMClient` với 2 cài đặt cụ thể:
1. **`MockLLMClient` (`model_name = "mock-grounded-extractive-v1"` – Mặc định Offline):**
   * Hoạt động hoàn toàn offline, không cần kết nối mạng hay API key, đảm bảo tính tái lập 100% khi chấm bài tự động trong Docker.
   * Cơ chế: Phân tích câu hỏi và các chunk trong prompt, chọn lọc các câu/mục chứa thông tin khớp trực tiếp với câu hỏi và đính kèm nguyên văn nhãn trích dẫn `[Nguồn: <source> / <section>]`.
2. **`OpenAICompatibleLLMClient` (Chế độ gọi LLM thật qua HTTP API):**
   * Kết nối tới mọi endpoint tương thích chuẩn OpenAI Chat Completions (`/v1/chat/completions`), mặc định dùng model `gpt-4o-mini` với `temperature = 0.0` để tối đa hóa tính nhất quán.
   * Cấu hình hoàn toàn qua biến môi trường (không lưu khóa bí mật trong mã nguồn):

| Biến môi trường | Tham số CLI | Giá trị mặc định | Mô tả |
|---|---|---|---|
| `LLM_PROVIDER` | `--provider` | `mock` | Chọn `mock`, `openai`, hoặc `auto` (tự động dùng `openai` nếu có `OPENAI_API_KEY`). |
| `LLM_MODEL` | `--model` | `gpt-4o-mini` | Tên model LLM sử dụng khi gọi API. |
| `OPENAI_API_KEY` | *(không)* | *(trống)* | Khóa API (không bao giờ được in ra log hay file output). |
| `OPENAI_BASE_URL` | *(không)* | `https://api.openai.com/v1` | Địa chỉ máy chủ API tương thích OpenAI. |
| `LLM_TIMEOUT` | *(không)* | `30` | Thời gian chờ tối đa (giây) cho mỗi lượt gọi API. |

   * **Cơ chế Fallback an toàn:** Nếu người dùng chọn `openai` nhưng quên đặt `OPENAI_API_KEY` hoặc gặp lỗi kết nối mạng, hệ thống tự động chuyển về `MockLLMClient` và ghi rõ vào header của file output: `mock-grounded-extractive-v1 [FALLBACK: ..., requested gpt-4o-mini]`.

### 10.2 Cách xây dựng Prompt và Context (`prompt_builder.py`)
Hàm `build_prompt(question, chunks, semantic=None, dialogue_state=None)` đóng gói thông tin thành cấu trúc chặt chẽ gồm 4 khối:
1. **Chỉ dẫn hệ thống (System Instructions):**
   * Chỉ được trả lời dựa trên thông tin nằm trong khối `CONTEXT` bên dưới.
   * Tuyệt đối không suy diễn hoặc bịa đặt ngày hạn nộp, thời gian, phòng học, tên giảng viên hay quy định không có trong `CONTEXT`.
   * Mọi câu khẳng định phải gắn kèm nhãn trích dẫn sao chép nguyên văn theo đúng định dạng `[Nguồn: <source> / <section>]`.
   * Nếu các đoạn `CONTEXT` không chứa đủ thông tin để trả lời câu hỏi, bắt buộc trả về đúng một chuỗi duy nhất: `INSUFFICIENT_CONTEXT`.
2. **Gợi ý ngữ nghĩa từ Phần I (Semantic Hint):**
   * Cung cấp `intent` và `entities` do Earley Parser trích xuất để định hướng trọng tâm câu trả lời, kèm ghi chú rõ: *(Gợi ý cấu trúc câu hỏi, không phải bằng chứng thay thế CONTEXT)*.
3. **Danh sách tài liệu truy xuất (Retrieved Context Blocks):**
   * Liệt kê từng chunk theo định dạng:
     ```text
     --- CONTEXT 1 [Nguồn: topics.txt / CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS] ---
     <nội dung văn bản của chunk>
     ```
4. **Câu hỏi đã giải tham chiếu (`resolved_query`):** Đặt ở cuối prompt để mô hình tập trung trả lời.

### 10.3 Cơ chế Answer Guard 3 lớp chống Hallucination (`answer_guard.py`, `assistant.py`)
Để giải quyết triệt để nguy cơ sinh ảo (*hallucination*) của LLM, hệ thống áp dụng 3 lớp kiểm duyệt độc lập:
* **Lớp 1 – Tiền kiểm tra trước khi gọi LLM (`check_context`):**
  1. Nếu `retrieval_status != "FOUND"` hoặc danh sách `chunks` rỗng $\rightarrow$ trả về ngay `NO_RELEVANT_CONTEXT`.
  2. Nếu văn phạm Phần I phân loại câu hỏi vào `intent == "UNKNOWN"` $\rightarrow$ trả về ngay `NO_RELEVANT_CONTEXT`.
  3. Nếu câu hỏi không khớp được bất kỳ intent hay entity nào (`UNPARSED` và `entity_ids` rỗng), hàm `lexical_support(question, chunks)` loại bỏ danh sách từ hư tiếng Việt (`_STOPWORDS`: *"là"*, *"gì"*, *"hôm"*, *"nay"*, *"thế"*, *"nào"*, ...) và tính tỷ lệ từ nội dung của câu hỏi xuất hiện trong các chunk. Nếu không có từ nội dung nào xuất hiện (`lexical_support <= 0.0`) $\rightarrow$ chặn ngay với `NO_RELEVANT_CONTEXT`.
* **Lớp 2 – Kiểm tra tính đầy đủ của thông tin & Sentinel (`INSUFFICIENT_CONTEXT`):**
  1. Trước và sau LLM, hệ thống kiểm tra các câu hỏi về thuộc tính mà KB chỉ chứa bản mẫu chưa có giá trị thực tế (ví dụ: hỏi **ngày giờ hạn nộp/deadline** chính thức của bài tập lớn trong khi `assignments.txt` chỉ ghi *"Lưu ý: dữ liệu mẫu, cần thay bằng lịch chính thức"*, hoặc hỏi **phòng học/địa điểm học**, hoặc hỏi chương vượt quá Chương 12 như `CH13`).
  2. Nếu mô hình trả về chuỗi sentinel `INSUFFICIENT_CONTEXT` $\rightarrow$ hệ thống trả trạng thái `INSUFFICIENT_CONTEXT` với thông báo lịch sự, không gắn nguồn giả.
* **Lớp 3 – Hậu kiểm tra căn cứ và xác thực thực thể số/ngày (`check_answer`):**
  1. **Xác thực nguồn trích dẫn (`Citation Validation`):** Trích xuất mọi nhãn `[Nguồn: source / section]` bằng biểu thức `_CITATION` và đối chiếu với tập các cặp `(source, section)` thực sự đã đưa vào prompt (`allowed`). Mọi trích dẫn bịa đặt (`invalid_citations`) bị xóa bỏ ngay lập tức; nếu không còn trích dẫn hợp lệ nào hoặc có trích dẫn bịa, trạng thái bị hạ xuống `NEEDS_REVIEW`.
  2. **Xác thực thông tin định lượng (`Fact Verification` – hàm `unsupported_facts`):**
     * Sử dụng biểu thức chính quy `_FACT` quét toàn bộ **ngày tháng** (`\d{1,2}[/-]\d{1,2}...`), **giờ** (`\d{1,2}[:h]\d{2}`), **địa chỉ email**, **URL** và **con số** (`\d+(?:[.,]\d+)?`) trong câu trả lời của mô hình (sau khi đã tách nhãn `[Nguồn: ...]`).
     * Chuẩn hóa các con số qua `_norm_fact` (bỏ số `0` ở đầu để `"tuần 5"` khớp hợp lệ với `"WEEK 05"`) và kiểm tra xem từng giá trị có xuất hiện trong văn bản gốc của các chunk đã gửi hay không.
     * Nếu phát hiện bất kỳ con số, ngày tháng hay URL nào do mô hình tự bịa ra (`unsupported_facts` khác rỗng), hệ thống **tịch thu câu trả lời**, gán trạng thái `NEEDS_REVIEW`, lưu câu gốc vào `raw_answer` để kiểm toán và trả thông báo an toàn cho người dùng:
       *"Câu trả lời do mô hình sinh ra chứa thông tin không có trong tài liệu truy xuất (...), nên hệ thống không đưa ra để tránh sai lệch."*

### 10.4 Xử lý ngữ cảnh hội thoại cục bộ – Phần III (`dialogue.py`)
Theo lý thuyết **Chương 7 (Discourse Situation)** và **Chương 12 (Discourse Context and Reference)**, ý nghĩa của một phát ngôn phụ thuộc vào trạng thái diễn ngôn hiện tại $D_{\text{initial}}$ và sau khi trả lời sẽ cập nhật thành trạng thái diễn ngôn mới $D_{\text{updated}}$.

1. **Cấu trúc trạng thái `DialogueState`:**
   ```python
   {
     "last_question": "Chương 6 nói về gì?",
     "last_intent": "GET_TOPIC",
     "last_entities": {"chapter": "CH06", "part": "PART_I", "topic": "POS_TAGGING"},
     "last_answer_entities": ["CH06", "POS_TAGGING", "PCFG"],
     "last_sources": [{"source": "topics.txt", "section": "CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS"}]
   }
   ```
   Ngoài ra, `DialogueState` lưu thêm `_ambiguous_candidates` khi câu hỏi trước nhắc tới nhiều thực thể cùng loại (ví dụ: `{"chapter": ["CH02", "CH03"]}`).

2. **Sáu quy tắc giải tham chiếu cục bộ (`resolve_reference` trong `dialogue.py`):**

| Loại tham chiếu| Dấu hiệu nhận diện trong câu hỏi | Quy tắc biến đổi sang query | Ví dụ thực tế  |
|---|---|---|---|
| `PRONOUN_IT` | Đại từ *"nó"* làm chủ ngữ/bổ ngữ | Thay *"nó"* bằng tên hiển thị chuẩn (`ENTITY_DISPLAY`) của thực thể trung tâm ở lượt trước (`chapter` $\succ$ `topic` $\succ$ `week` $\succ$ `lo`). | Lượt trước: *"Chương 6 nói về gì?"* (`CH06`)<br>Câu hỏi: *"Nó có liên quan đến gán nhãn từ loại không?"*<br>$\rightarrow$ `"Chương 6 có liên quan đến gán nhãn từ loại không?"` |
| `THIS_CHAPTER` | Cụm chỉ xuất *"chương này"*, *"chương đó"* | Thay bằng `"Chương X"` lấy từ `last_entities["chapter"]`. | Lượt trước: `CH06`<br>Câu hỏi: *"Chương này học vào tuần nào?"*<br>$\rightarrow$ `"Chương 6 học vào tuần nào?"` |
| `NEXT_CHAPTER` | Câu hỏi rút gọn *"Còn chương sau?"*, *"Thế còn chương tiếp theo?"* | Tăng chỉ số chương hiện tại lên $X + 1$ (`CH06` $\rightarrow$ `CH07`) và kế thừa kiểu câu hỏi của lượt trước. Nếu $X + 1 > 12$ (`CH13`), đánh dấu vượt phạm vi KB. | Lượt trước: *"Chương 8 học vào tuần nào?"* (`CH08`)<br>Câu hỏi: *"Còn chương sau?"*<br>$\rightarrow$ `"Chương 9 học vào tuần nào?"` |
| `THIS_CONTENT` | Cụm chỉ xuất *"nội dung này"*, *"chủ đề này"* | Thay bằng thực thể chương/chủ đề/L.O. vừa thảo luận ở lượt trước. | Lượt trước: `CH09`<br>Câu hỏi: *"Nội dung này học vào tuần nào?"*<br>$\rightarrow$ `"Chương 9 học vào tuần nào?"` |
| `THIS_PART` | Cụm chỉ xuất *"phần này"*, *"phần đó"* (khi đang hỏi về bài tập lớn) | Thay bằng `"Phần I/II/III/IV"` hoặc `"bài tập lớn"` từ `last_entities`. | Lượt trước: *"Bài tập lớn có phần hội thoại không?"*<br>Câu hỏi: *"Phần này cần xuất file gì?"*<br>$\rightarrow$ `"bài tập lớn cần xuất file gì?"` |
| `RESOURCE_OF_THIS_PART` | *"Tài liệu của phần này/chương này ở đâu / là gì?"* | Kết hợp thực thể chính và các chủ đề trong câu trả lời lượt trước (`last_answer_entities`) thành truy vấn tài liệu đầy đủ. | Lượt trước: *"Tuần 2 học những nội dung gì?"* (`WEEK_02`, `CH02`)<br>Câu hỏi: *"Tài liệu của phần này ở đâu?"*<br>$\rightarrow$ `"Tài liệu tham khảo cho Tuần 2 (Phần I, Chương 2) là gì?"` |

3. **Xử lý thiếu tiền từ hoặc tiền từ mơ hồ (`NEED_CLARIFICATION`):**
   * **Thiếu tiền từ (Missing Antecedent):** Khi bắt đầu cuộc hội thoại mới hoặc vừa gõ lệnh `RESET` (`DialogueState.is_empty() == True`) mà người dùng hỏi ngay *"Nó học vào tuần nào?"* hoặc *"Còn chương sau?"*, hệ thống không đoán mò mà trả về `status = "NEED_CLARIFICATION"` yêu cầu người dùng nêu rõ chương/chủ đề muốn hỏi.
   * **Tiền từ mơ hồ (Ambiguous Antecedent):** Khi lượt trước người dùng hỏi so sánh 2 chương (*"Chương 2 và Chương 3 khác nhau như thế nào?"*), `_ambiguous_candidates["chapter"] = ["CH02", "CH03"]`. Ở lượt tiếp theo nếu người dùng hỏi *"Chương này học vào tuần nào?"*, hệ thống phát hiện sự nhập nhằng tham chiếu và trả về `status = "NEED_CLARIFICATION"`: *"Trong lượt trước có nhiều hơn một chương (Chương 2, Chương 3). Bạn muốn hỏi về Chương 2 hay Chương 3?"*.

---

## 11. Mô tả cách đánh giá

---

## 12. Các hạn chế của hệ thống

1. **Tính cứng nhắc của văn phạm CFG biên soạn thủ công:**
   * Văn phạm phi ngữ cảnh ở Phần I đảm bảo độ chính xác cú pháp và ngữ nghĩa 100% cho các mẫu câu đã định nghĩa, nhưng nếu người dùng sử dụng câu khẩu ngữ rút gọn, đảo trật tự từ tự do, viết tắt hoặc sai dấu tiếng Việt (ngoài danh sách alias), Earley Parser sẽ trả về `()` (`UNPARSED`).
2. **Giới hạn biểu diễn từ vựng của TF-IDF:**
   * Tầng truy xuất dựa trên TF-IDF kết hợp từ điển đồng nghĩa tĩnh (`data/entities.txt`). Nếu người dùng diễn đạt bằng các cụm từ đồng nghĩa hoặc cách mô tả khái niệm hoàn toàn mới chưa có trong `entities.txt` (không chia sẻ từ vựng bề mặt với tài liệu), điểm `LexicalScore` sẽ thấp. Có thể khắc phục trong tương lai bằng cách kết hợp thêm Dense Embedding (Semantic Search đa ngôn ngữ).
3. **Sự khác biệt giữa `MockLLMClient` (Offline) và LLM thực thụ (Online):**
   * `MockLLMClient` đảm bảo hệ thống chạy độc lập, cực nhanh và tái lập 100% kết quả mà không cần API key, nhưng câu trả lời có tính chất trích xuất nguyên đoạn (*extractive*) nên đôi khi dài hơn mức cần thiết và không tự suy luận để từ chối các câu ngoài miền có trùng từ vựng ngẫu nhiên như khi chạy với `OpenAICompatibleLLMClient`.
4. **Phạm vi của bộ kiểm tra thực thể trong `answer_guard.py`:**
   * Hậu kiểm tra `unsupported_facts` phát hiện chính xác 100% các hành vi bịa đặt con số, ngày tháng, giờ giấc, email, URL và trích dẫn nguồn sai. Tuy nhiên, nếu một mô hình ngôn ngữ giữ nguyên các con số trong nguồn nhưng đảo ngược ý nghĩa mệnh đề bằng chữ (ví dụ đổi *"bắt buộc"* thành *"không bắt buộc"*), bộ lọc biểu thức chính quy chưa thể tự phát hiện mà cần tới mô hình Natural Language Inference (NLI).
5. **Giới hạn của quản lý ngữ cảnh hội thoại cục bộ 1 bước (`DialogueState`):**
   * Mô-đun `dialogue.py` được thiết kế tối ưu cho các tham chiếu cục bộ liền kề (1-step anaphora & ellipsis resolution) theo đúng yêu cầu Phần III của đề bài. Hệ thống chưa hỗ trợ xếp chồng nhiều chủ đề hội thoại (dialogue stack) hoặc quay lại tham chiếu một thực thể đã nhắc tới cách đó 3–4 lượt hội thoại trước.
=======
# BÀI TẬP LỚN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (CO3085)
## Trợ lý hỗ trợ học tập thông minh – Course Assistant

---

## 1. Thành viên nhóm
* **Môn học:** Xử lý ngôn ngữ tự nhiên (CO3085) – Học kỳ HK261
* **Khoa:** Khoa Khoa học và Kỹ thuật Máy tính – Trường Đại học Bách khoa – ĐHQG TP.HCM
* **Danh sách thành viên:**
  - Sinh viên 1: `Nguyễn Văn Bảo` – MSSV: `2433037`
  - Sinh viên 2: `Phạm Xuân Bắc` – MSSV: `2310280`
  - Sinh viên 3: `Trần Lê Đức An` – MSSV: `2310038`
  - Sinh viên 4: `[Họ và tên]` – MSSV: `[MSSV4]`

---

## 2. Hướng dẫn cài đặt

### 2.1 Yêu cầu hệ thống
* **Python:** `>= 3.8` (khuyến nghị Python `3.10+`, đã kiểm thử ổn định trên Python `3.12`)
* **Hệ điều hành:** Windows / Linux / macOS
* **Docker:** Dockerfile hiện tại chưa cấu hình entrypoint để chạy Course Assistant; dùng cài đặt pip.

### 2.2 Cài đặt trực tiếp bằng `pip`
```bash
cd python
pip install -r requirements.txt
pip install -e .
```

### 2.3 Docker (chưa sẵn sàng cho Course Assistant)

Dockerfile hiện tại vẫn dùng entrypoint mẫu `hcommon.cli` (`python3 -m hcommon.cli ...`), không khởi chạy package `nlp.app`. Vì vậy chưa nên dùng lệnh `docker build/run` này như cách chạy Course Assistant. Hãy dùng cài đặt pip ở mục 2.2; chỉ bổ sung hướng dẫn Docker sau khi Dockerfile được cập nhật và kiểm thử end-to-end.

---

## 3. Hướng dẫn chạy chương trình

Tất cả các lệnh dưới đây được thực thi từ thư mục `python/`:

```bash
# 1. Sinh câu mẫu hợp lệ từ văn phạm (Phần I - 3.2, tối đa 10.000 câu) -> nlp/app/output/samples.txt
python -m nlp.app.utils

# 2. Phân tích cú pháp tập câu kiểm thử (Phần I - 3.3) -> nlp/app/output/parse-results.txt
python -m nlp.app.run_parser \
  --input nlp/app/input/sentences.txt \
  --output nlp/app/output/parse-results.txt \
  --grammar nlp/app/output/grammar.txt

# 3. Truy xuất Knowledge Base độc lập (Phần II - 4.2) -> nlp/app/output/retrieval.txt
python -m nlp.app.run_retrieval

# 4. Chạy toàn bộ pipeline Course Assistant (Phần II LLM/RAG, Phần III Hội thoại, Phần IV Đánh giá)
#    Tự động xuất dialogue.txt, answer.txt, evaluation.txt vào nlp/app/output/ và ../output/
python -m nlp.app.run_assistant --data-dir data

# 5. Chạy hỏi đáp 1 câu đơn lẻ hoặc chế độ hội thoại tương tác nhiều lượt
python -m nlp.app.run_assistant --data-dir data --question "Chương 6 nói về gì?" --json
python -m nlp.app.run_assistant --data-dir data --interactive

# 6. Khởi động HTTP API (/health, /retrieve, /answer, /reset)
python -m nlp.app.api --host 127.0.0.1 --port 8000

# 7. Chạy bộ kiểm thử tự động (unittest)
python -m unittest discover -s test -v
```

---

## 4. Mô tả kiến trúc hệ thống

### 4.1 Sơ đồ kiến trúc tổng thể (End-to-End Pipeline)
Hệ thống được thiết kế theo kiến trúc phân tầng mô-đun (*modular layered pipeline*), giữ nguyên vai trò nền tảng của văn phạm và phân tích cú pháp ở Phần I, đồng thời tích hợp mượt mà với cơ chế truy xuất lai (*Hybrid Retrieval*), kiểm soát căn cứ (*Answer Guard*) của nhánh **LLM/RAG (Phần II)** và quản lý **ngữ cảnh hội thoại cục bộ (Phần III)**:

```text
====================================================================================================
                        KIẾN TRÚC PIPELINE HỆ THỐNG COURSE ASSISTANT
====================================================================================================

  [ Câu hỏi người dùng (Natural Language Input) ] + [ DialogueState (Ngữ cảnh lượt trước) ]
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 1: XỬ LÝ NGỮ CẢNH HỘI THOẠI CỤC BỘ (Phần III - nlp/app/dialogue.py)                 │
  │ - Phát hiện từ chỉ xuất / đại từ: "nó", "chương này", "phần này", "còn chương sau?"...   │
  │ - Nếu thiếu tiền từ (sau RESET) hoặc tiền từ mơ hồ (nhiều chương) -> NEED_CLARIFICATION  │
  │ - Nếu giải tham chiếu thành công -> Sinh câu hỏi đầy đủ: resolved_query                  │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ resolved_query
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 2: PHÂN TÍCH CÚ PHÁP CFG (Phần I - nlp/app/parser.py & output/grammar.txt)          │
  │ - Chuẩn hóa Unicode NFC, khớp cụm từ đa từ (multi-word terminals) không phân biệt hoa/thường│
  │ - Giải thuật Earley Chart Parser (Predictor, Scanner, Completer)                         │
  │ - Đầu ra: Cây cú pháp ParseNode dạng ngoặc (S (Q_... ...)) hoặc () nếu ngoài văn phạm   │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ ParseNode / None
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 3: DIỄN GIẢI NGỮ NGHĨA & INTENT/ENTITY (Phần II - semantic.py & semantic_frame.py)  │
  │ - Duyệt cây cú pháp theo nguyên lý thành phần (Compositionality)                         │
  │ - Ánh xạ nút gốc Q_* -> Intent (GET_COURSE_INFO, GET_TOPIC, GET_SCHEDULE, GET_LO, ...)   │
  │ - Chuẩn hóa thực thể qua từ điển đồng nghĩa data/entities.txt -> entity_ids & slots      │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ SemanticFrame {intent, entities, entity_ids, parse_status}
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 4: TRUY XUẤT TRI THỨC LAI (Phần II - knowledge_base.py & retriever.py)              │
  │ - Phân đoạn 7 file dữ liệu gốc (data/*.txt) thành các chunk gắn nhãn nguồn & entity_ids  │
  │ - Mở rộng truy vấn (Query Expansion) bằng alias + token thực thể + từ khóa intent        │
  │ - Tính điểm: CombinedScore = 0.75 * Cosine_TFIDF + 0.25 * SemanticOverlap                │
  │ - Chọn top_k = 3 chunk có điểm >= min_score (0.08) -> FOUND hoặc NO_RELEVANT_CONTEXT     │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ Retrieved Chunks + Scores
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 5: TIỀN KIỂM TRA & XÂY DỰNG PROMPT (answer_guard.py & prompt_builder.py)            │
  │ - Pre-LLM Guard (check_context): Chặn câu UNKNOWN hoặc không có từ vựng chung với chunk  │
  │ - Kiểm tra thuộc tính thiếu trong KB (ngày deadline chính thức, phòng học)               │
  │ - Đóng gói System Rules + Semantic Hint + Context Blocks có nhãn [Nguồn: file / section] │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │ Grounded Prompt
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 6: SINH CÂU TRẢ LỜI & HẬU KIỂM TRA (llm_client.py & answer_guard.py)                │
  │ - Gọi MockLLMClient (offline) hoặc OpenAICompatibleLLMClient (online API)                │
  │ - Post-LLM Guard (check_answer):                                                         │
  │   (1) Bắt sentinel INSUFFICIENT_CONTEXT                                                  │
  │   (2) Xác thực mọi trích dẫn [Nguồn: ...] phải nằm trong các chunk đã gửi                │
  │   (3) Kiểm tra mọi số, ngày, giờ, email, URL phải tồn tại trong văn bản chunk            │
  │ - Cập nhật DialogueState cho lượt hội thoại tiếp theo                                    │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  [ Kết quả đầu ra: output/parse-results.txt, retrieval.txt, answer.txt, dialogue.txt, evaluation.txt ]
```

---

## 5. Mô tả Grammar

### 5.1 Cơ sở lý thuyết: Văn phạm phi ngữ cảnh (CFG) và Văn phạm ngữ nghĩa (Semantic Grammar)
1. **Văn phạm phi ngữ cảnh (CFG):**
   Một văn phạm phi ngữ cảnh được định nghĩa bởi bộ 4 thành phần $G = (V, \Sigma, R, S)$, trong đó:
   * $V$: Tập hữu hạn các ký hiệu phi kết thúc (*Non-terminals*, viết trong cặp dấu `<...>`).
   * $\Sigma$: Tập hữu hạn các ký hiệu kết thúc (*Terminals*, các từ hoặc cụm từ tiếng Việt/Anh đặt trong dấu `"..."`), rời nhau với $V$.
   * $R$: Tập các luật sản xuất (*Production rules*) có dạng $A \rightarrow \alpha$, với $A \in V$ và $\alpha \in (V \cup \Sigma)^*$.
   * $S \in V$: Ký hiệu bắt đầu đại diện cho một câu hoàn chỉnh.
2. **Văn phạm ngữ nghĩa (Semantic Grammar):**
    (*Other Strategies for Semantic Interpretation*), đối với các hệ thống hỏi đáp trong miền ứng dụng hẹp (*restricted-domain QA*), thay vì chỉ sử dụng các phạm trù thuần cú pháp (`NP`, `VP`, `PP`), ta thiết kế các ký hiệu phi kết thúc mang **ý nghĩa miền bài toán** (như `<Q_COURSE_INFO>`, `<Q_SCHEDULE>`, `<CHAPTER_REF>`, `<TOPIC_REF>`, `<LO_REF>`). Cách tiếp cận này giúp:
   * Giảm đáng kể độ nhập nhằng cấu trúc (*structural ambiguity*);
   * Ràng buộc trực tiếp các giới hạn lựa chọn (*selectional restrictions*): ví dụ động từ/cụm vị ngữ hỏi thời lượng thi (`<EXAM_TAIL> ::= "bao nhiêu phút?"`) chỉ kết hợp với `<EXAM_REF>` (`midterm`, `final`), còn cụm vị ngữ hỏi tín chỉ chỉ kết hợp với `<COURSE_REF>`;
   * Cho phép suy ra trực tiếp **Logical Form / Semantic Frame** từ chính cấu trúc cây cú pháp mà không cần tầng biến đổi trung gian phức tạp.

### 5.2 Thiết kế chi tiết tập luật trong `output/grammar.txt`
File văn phạm `output/grammar.txt` (được sinh và quản lý bởi `course_grammar.py` và `grammar_file.py`) định nghĩa ký hiệu gốc `<S>` phân nhánh thành **9 phạm trù câu hỏi cấp cao**:

```bnf
<S> ::= <Q_COURSE_INFO>
      | <Q_SCHEDULE>
      | <Q_TOPIC>
      | <Q_ASSIGNMENT>
      | <Q_DEADLINE>
      | <Q_RESOURCE>
      | <Q_RULE>
      | <Q_LO>
      | <Q_UNKNOWN>
```

Chi tiết từng nhóm luật trong văn phạm:
1. **Nhóm hỏi thông tin chung môn học (`<Q_COURSE_INFO>`):**
   * Hỗ trợ hỏi số tín chỉ, mã môn học, mô tả môn học, cấu trúc các phần và thời lượng bài thi giữa kỳ/cuối kỳ, có hoặc không có tiền tố lịch sự (`<POLITE_PREFIX>`):
   ```bnf
   <Q_COURSE_INFO> ::= <COURSE_REF> <COURSE_INFO_TAIL>
                     | <POLITE_PREFIX> <COURSE_REF> <COURSE_INFO_TAIL>
                     | <EXAM_REF> <EXAM_TAIL>
                     | <POLITE_PREFIX> <EXAM_REF> <EXAM_TAIL>
   <COURSE_REF>    ::= "Môn NLP" | "môn học" | "CO3085" | "C03085" | "môn CO3085" | "môn C03085" | "Xử lý Ngôn ngữ Tự nhiên"
   <EXAM_REF>      ::= "Midterm" | "midterm" | "Final" | "final" | "thi giữa kỳ" | "thi cuối kỳ"
   <EXAM_TAIL>     ::= "bao nhiêu phút?" | "kéo dài bao nhiêu phút?" | "thi bao lâu?"
   ```
2. **Nhóm hỏi lịch học và kế hoạch giảng dạy (`<Q_SCHEDULE>`):**
   * Hỗ trợ hỏi nội dung theo tuần (`<WEEK_REF>`), hỏi tuần học của một chương (`<CHAPTER_REF>`) hoặc một chủ đề (`<TOPIC_REF>`), và hỏi lịch học chung:
   ```bnf
   <Q_SCHEDULE> ::= <WEEK_REF> <SCHEDULE_WEEK_TAIL>
                  | <CHAPTER_REF> <SCHEDULE_CHAPTER_TAIL>
                  | <SCHEDULE_TOPIC_REF> <SCHEDULE_TOPIC_TAIL>
                  | "Khi nào học" <TOPIC_REF> "?"
                  | <DEICTIC_CHAPTER> <SCHEDULE_CHAPTER_TAIL>
                  | <DEICTIC_CONTENT> <SCHEDULE_CHAPTER_TAIL>
                  | <SCHEDULE_WHEN>
   ```
3. **Nhóm hỏi nội dung và chủ đề bài học (`<Q_TOPIC>`):**
   * Hỗ trợ hỏi nội dung một chương (`Chương 6 nói về gì?`), tìm chương chứa chủ đề (`Chương nào nói về PCFG?`, `Chương nào đề cập đến gán nhãn từ loại?`), kiểm tra mối liên hệ giữa chương và chủ đề (`Chương 6 có liên quan đến gán nhãn từ loại không?`), so sánh hai chương (`Chương 2 và Chương 3 khác nhau như thế nào?`) và yêu cầu giải thích/giới thiệu chủ đề (`giới thiệu về mạng ngữ nghĩa`):
   ```bnf
   <Q_TOPIC> ::= <CHAPTER_REF> <TOPIC_CHAPTER_TAIL>
               | "Chương nào nói về" <TOPIC_REF> "?"
               | "Chương nào đề cập đến" <TOPIC_REF> "?"
               | "Chương nào học về" <TOPIC_REF> "?"
               | <CHAPTER_REF> <TOPIC_RELATED_TAIL>
               | <DEICTIC_CHAPTER> <TOPIC_RELATED_TAIL>
               | <CHAPTER_REF> "và" <CHAPTER_REF> <TOPIC_COMPARE_TAIL>
               | "giải thích" <TOPIC_REF> | "giới thiệu về" <TOPIC_REF>
   ```
4. **Nhóm hỏi thông tin bài tập lớn (`<Q_ASSIGNMENT>`) và hạn nộp (`<Q_DEADLINE>`):**
   * Bao phủ câu hỏi về yêu cầu chung của bài tập lớn, nội dung từng phần (`Phần I` – `Phần IV`), câu hỏi Yes/No về thành phần bài tập (`Bài tập lớn có phần hội thoại không?`), trách nhiệm nộp bài của sinh viên (`Tôi cần hoàn thành những bài tập nào?`, `Sinh viên phải nộp những gì?`) và thời hạn nộp bài (`Deadline của bài tập lớn là khi nào?`, `Hạn nộp bài tập lớn là khi nào?`).
5. **Nhóm hỏi tài liệu tham khảo (`<Q_RESOURCE>`):**
   * Hỗ trợ tìm tài liệu theo chủ đề (`Tài liệu nào nói về PCFG?`, `Tài liệu nào liên quan đến đồng tham chiếu?`), theo phạm vi chương/tuần/phần (`Tài liệu tham khảo cho Chương 6 là gì?`, `Tài liệu cho Phần I ở đâu?`), theo nhánh chuyên ngành (`statistical NLP`, `Python NLP`, `machine translation`) và câu mệnh lệnh (`liệt kê tài liệu tham khảo`, `tìm tài liệu về semantic grammar`).
6. **Nhóm hỏi quy định môn học (`<Q_RULE>`) và chuẩn đầu ra (`<Q_LO>`):**
   * `<Q_RULE>` kiểm tra quy định sử dụng công cụ AI, mở rộng LLM/RAG, thư viện ngoài và trích dẫn nguồn.
   * `<Q_LO>` truy vấn nội dung và năng lực yêu cầu của các chuẩn đầu ra `L.O.2.5`, `L.O.3.3`, `L.O.4.2`.
7. **Nhóm câu hỏi ngoài miền mẫu (`<Q_UNKNOWN>`):**
   * Nhận diện trực tiếp các mẫu câu ngoài phạm vi môn học để kiểm thử tầng lọc ngữ nghĩa.

### 5.3 Giải thuật sinh câu từ văn phạm (`sentence_generator.py`, `utils.py`)
* **Nguyên lý:** Trong `SentenceGenerator.generate()`, quá trình sinh câu được mô hình hóa như bài toán duyệt đồ thị các dẫn xuất trái nhất (*leftmost derivations*) theo **chiều rộng (Breadth-First Search – BFS)** sử dụng cấu trúc hàng đợi `collections.deque`:
  1. Khởi tạo hàng đợi `pending = deque([([grammar.start_symbol], 0)])`, danh sách kết quả `sentences = []` và tập khử trùng lặp `seen = set()`.
  2. Ở mỗi bước lặp (khi `pending` còn phần tử và `len(sentences) < max_sentences`):
     * Lấy ra trạng thái `(symbols, expansion_count)` ở đầu hàng đợi.
     * Tìm vị trí `index` của ký hiệu phi kết thúc đầu tiên từ trái sang phải (`grammar.is_nonterminal(symbol)`).
     * **Trường hợp cơ sở (`index is None`):** Mọi ký hiệu trong `symbols` đều là ký hiệu kết thúc (*terminals*). Ghép các token bằng khoảng trắng, chuẩn hóa khoảng trắng trước dấu câu (`re.sub(r"\s+([,.;:!?])", r"\1", sentence)`), và nếu câu chưa xuất hiện trong `seen` thì thêm vào `sentences`.
     * **Bước mở rộng (`index is not None`):** Nếu `expansion_count < max_expansions` (mặc định `30` để chặn đệ quy vô hạn), duyệt qua mọi vế phải `production` của `symbols[index]` và đẩy dãy ký hiệu mới `symbols[:index] + list(production) + symbols[index + 1:]` kèm `expansion_count + 1` vào cuối hàng đợi `pending`.
* **Ưu điểm của BFS:** Đảm bảo sinh tất cả các câu ngắn, cấu trúc cơ bản và bao phủ đồng đều mọi nhánh `<Q_*>` trước khi mở rộng sang các biến thể dài có tiền tố lịch sự.
* **Kết quả thực thi:** Sinh ra **8.926 câu** hợp lệ, không trùng lặp (nằm trong giới hạn $\le 10.000$ câu của đề bài) và lưu tại `output/samples.txt`.

---

## 6. Mô tả Parser

### 6.1 Cơ sở lý thuyết: Chart Parsing và Giải thuật Earley
* Trong bài giảng Chương 3 (*Grammar and Parsing*), các giải thuật phân tích cú pháp quay lui đơn thuần (*Simple Top-Down / Bottom-Up Parser*) gặp nhược điểm lớn là phải phân tích đi phân tích lại cùng một cụm từ con khi một nhánh giả thuyết thất bại.
* **Chart Parsing** giảm nhược điểm này bằng cách sử dụng cấu trúc bảng **Chart** lưu các cung hoạt động (*active arcs / dotted rules*) và các thành phần đã hoàn tất (*completed constituents*).
* Hệ thống cài đặt **Earley Chart Parser** (`CourseParser` trong `nlp/app/parser.py`), hoạt động trực tiếp trên mức ký tự/cụm từ của câu đã chuẩn hóa mà không đòi hỏi phải chuyển văn phạm về dạng chuẩn Chomsky (CNF).

### 6.2 Cấu trúc dữ liệu và Giải thuật chi tiết trong `parser.py`
Cho câu đầu vào `sentence`, sau khi chuẩn hóa Unicode dựng sẵn `text = unicodedata.normalize("NFC", sentence.strip())` có độ dài $N = \text{len(text)}$:
* **Biểu diễn trạng thái Earley (Dotted Rule):** Mỗi trạng thái là một bộ 5 phần tử:
  $$\text{state} = (\text{lhs}, \text{prod}, \text{dot}, \text{origin}, \text{children})$$
  tương ứng với luật $\text{lhs} \rightarrow \alpha \bullet \beta$ bắt đầu tại vị trí ký tự $\text{origin}$, đã khớp được $\text{dot}$ ký hiệu trong $\text{prod}$ và tích lũy danh sách các nút con $\text{children}$.
* **Bảng `chart` và hàng đợi `agenda`:** Khởi tạo `chart = [set() for _ in range(N + 1)]` (dùng `set` để kiểm tra trùng lặp trạng thái trong $O(1)$) và `agenda = [[] for _ in range(N + 1)]` (duyệt tuần tự các trạng thái tại mỗi vị trí). Hàm `add(pos, ...)` tự động bỏ qua khoảng trắng (`_skip_spaces`) trước khi chèn trạng thái vào `chart[pos]` và `agenda[pos]`.
* **Khởi tạo:** Thêm luật giả khởi đầu `("__START__", (("N", "S"),), dot=0, origin=0, children=())` vào vị trí `pos = 0`.
* **Vòng lặp chính ($0 \le \text{pos} \le N$):** Duyệt từng trạng thái trong `agenda[pos]` và áp dụng 1 trong 3 phép toán:
  1. **Predictor (Dự đoán – khi ký hiệu sau dấu chấm là Non-terminal `("N", sym)`):**
     * Với mọi luật con `child_prod` của `sym`, thêm trạng thái `(sym, child_prod, 0, pos, ())` vào `chart[pos]`.
     * *Xử lý luật rỗng (Nullable `<EMPTY>`):* Nếu tại chính vị trí `pos` đã có trạng thái của `sym` hoàn tất tức thời (`d_orig == pos` và `d_dot == len(d_prod)`), tiến ngay dấu chấm của trạng thái hiện tại lên `dot + 1`.
  2. **Scanner (Quét từ vựng – khi ký hiệu sau dấu chấm là Terminal `("T", sym)`):**
     * Kiểm tra đoạn văn bản `text[pos : pos + len(sym)]` có khớp không phân biệt hoa thường (`casefold()`) với `sym` hay không.
     * Kiểm tra điều kiện biên từ (*word-boundary check*): vị trí kết thúc `m_end = pos + len(sym)` phải nằm ở cuối câu, hoặc theo sau bởi khoảng trắng, hoặc theo sau bởi dấu câu `?!,.;:` (ngăn lỗi khớp tiền tố như khớp `"môn"` vào giữa một từ dài).
     * Nếu khớp, thêm trạng thái mới tại vị trí `m_end` với `dot + 1` và gắn nút lá `ParseNode(sym)` vào `children`.
  3. **Completer (Hoàn tất – khi `dot == len(prod)`):**
     * Tạo nút cú pháp hoàn chỉnh `node = ParseNode(lhs, children)`.
     * Duyệt các trạng thái cha `(p_lhs, p_prod, p_dot, p_orig, p_ch)` trong `chart[origin]`; nếu trạng thái cha đang chờ đúng `("N", lhs)` tại `p_dot`, thêm trạng thái cha đã tiến dấu chấm `p_dot + 1` kèm `p_ch + (node,)` vào `chart[pos]`.
* **Điều kiện chấp nhận và Đầu ra:**
  * Sau khi duyệt tới `final_pos = _skip_spaces(text, N)`, nếu trong `agenda[final_pos]` tồn tại trạng thái `lhs == "__START__"`, `dot == 1`, `orig == 0`, trả về cây `ParseNode("S", ...)` và xuất chuỗi ngoặc đơn qua `node.to_bracketed()`.
  * Ngược lại, trả về `None` và hàm `parse_line()` xuất đúng ký hiệu `()` theo yêu cầu mục 3.3.
  * Hàm `parse()` được bọc bởi `@lru_cache(maxsize=50000)` giúp tăng tốc độ phân tích hàng loạt khi chạy đánh giá trên hàng nghìn mẫu.

### 6.3 Ví dụ kết quả phân tích cú pháp (`output/parse-results.txt`)
* **Câu hợp lệ (thuộc văn phạm):**
  * Đầu vào: `Cho hỏi CO3085 có bao nhiêu tín chỉ?`
  * Đầu ra: `(S (Q_COURSE_INFO (POLITE_PREFIX (cho hỏi)) (COURSE_REF (CO3085)) (COURSE_INFO_TAIL (có bao nhiêu tín chỉ?))))`
  * Đầu vào: `Chương 6 có liên quan đến gán nhãn từ loại không?`
  * Đầu ra: `(S (Q_TOPIC (CHAPTER_REF (chương) (CHAPTER_NUMBER (6))) (TOPIC_RELATED_TAIL (có liên quan đến) (TOPIC_REF (gán nhãn từ loại)) (không?))))`
* **Câu không hợp lệ / ngoài văn phạm:**
  * Đầu vào: `Thời tiết hôm nay thế nào?`
  * Đầu ra: `()`

---

## 7. Mô tả Semantic Representation

### 7.1 Cơ sở lý thuyết: Dạng luận lý (Logical Form) và Nguyên lý thành phần (Compositionality)
* **Logical Form (Chương 7):** Biểu diễn ngữ nghĩa độc lập ngữ cảnh (*context-independent meaning representation*) đóng vai trò cầu nối giữa phân tích cú pháp và cơ sở tri thức. Đối với các câu hỏi trong hệ thống hỏi đáp, dạng luận lý xác định rõ **tác tử truy vấn (Query Operator / Intent)** và **các đối số thực thể được ràng buộc (Bound Arguments / Thematic Slots)**.
* **Principle of Compositionality (Chương 8):** Nghĩa của một cấu trúc phức hợp được tổng hợp từ nghĩa của các thành phần con và quy tắc kết hợp cú pháp của chúng. Nhờ thiết kế Semantic Grammar ở Phần I, cấu trúc phân cấp của `ParseNode` cho phép tổng hợp khung ngữ nghĩa một cách tất định.

### 7.2 Cấu trúc `SemanticFrame` và Biểu diễn vị từ (`semantic.py`, `semantic_frame.py`)
Trong hệ thống, mỗi câu hỏi sau khi phân tích cú pháp được biểu diễn dưới hai dạng tương đương:
1. **Biểu diễn vị từ tường minh (Predicate-Argument Representation):**
   Có dạng `INTENT_PREDICATE(slot_1=ENTITY_1, slot_2=ENTITY_2, ...)`:
   * `GET_CREDIT(course=CO3085)` hoặc `GET_COURSE_INFO(exam=MIDTERM)`
   * `GET_SCHEDULE(week=WEEK_02)` hoặc `GET_SCHEDULE(chapter=CH06)` hoặc `GET_SCHEDULE(topic=DEPENDENCY_GRAMMAR)`
   * `GET_TOPIC(chapter=CH06)` hoặc `GET_TOPIC(chapter=CH06, topic=POS_TAGGING)` hoặc `GET_TOPIC(chapter_1=CH02, chapter_2=CH03)`
   * `GET_ASSIGNMENT(part=PART_II)` hoặc `GET_ASSIGNMENT(assignment=BTL01)`
   * `GET_DEADLINE(assignment=BTL01)`
   * `GET_RESOURCE(topic=PCFG)` hoặc `GET_RESOURCE(chapter=CH06)` hoặc `GET_RESOURCE(week=WEEK_08)`
   * `GET_RULE(rule=AI_USAGE)` hoặc `GET_RULE(rule=LLM_RAG)`
   * `GET_LO(lo=LO3.3)`
2. **Cấu trúc `SemanticFrame` (Dictionary chuẩn giao tiếp giữa các tầng):**
   ```json
   {
     "intent": "GET_TOPIC",
     "entities": {
       "chapter": "CH06",
       "topic": "POS_TAGGING"
     },
     "entity_ids": ["CH06", "POS_TAGGING"],
     "parse_status": "PARSED"
   }
   ```
   * Khi câu đầu vào không thuộc văn phạm (`parse_tree is None`), hàm `unparsed_frame()` trong `semantic.py` trả về khung trung lập `{"intent": None, "entities": {}, "entity_ids": [], "parse_status": "UNPARSED"}` (tuyệt đối không đoán mò intent sai lệch).

### 7.3 Khung ngữ nghĩa phía tài liệu (`chunk_frame`)
Không chỉ câu hỏi mà mỗi đoạn tài liệu (chunk) trong Knowledge Base cũng được gắn một khung ngữ nghĩa tại thời điểm lập chỉ mục qua hàm `chunk_frame(document)` trong `semantic_frame.py`:
```python
KIND_TO_INTENTS = {
    "course_info": {"GET_COURSE_INFO", "GET_LO"},
    "schedule":    {"GET_SCHEDULE"},
    "topics":      {"GET_TOPIC"},
    "assignments": {"GET_ASSIGNMENT", "GET_DEADLINE"},
    "resources":   {"GET_RESOURCE"},
    "regulations": {"GET_RULE"},
    "faq":         {"GET_COURSE_INFO", "GET_SCHEDULE", "GET_TOPIC", "GET_ASSIGNMENT",
                    "GET_DEADLINE", "GET_RESOURCE", "GET_RULE", "GET_LO"},
}
```
Việc biểu diễn cả câu hỏi lẫn tài liệu trong cùng một không gian `SemanticFrame` (`intents` + `entity_ids`) cho phép hệ thống thực hiện đối sánh ngữ nghĩa chính xác ở tầng Retriever.

---

## 8. Mô tả phương pháp xử lý Intent/Entity

### 8.1 Phương pháp nhận diện Intent (`semantic_frame.py`)
* **Ánh xạ từ nút cú pháp cấp câu (`TREE_INTENT_TO_INTENT`):**
  Hàm `frame_from_tree(tree, query, aliases)` tìm nút con trực tiếp của `S` có nhãn bắt đầu bằng `"Q_"` và tra bảng ánh xạ 1–1:
  - `Q_COURSE_INFO` $\rightarrow$ `GET_COURSE_INFO`
  - `Q_SCHEDULE` $\rightarrow$ `GET_SCHEDULE`
  - `Q_TOPIC` $\rightarrow$ `GET_TOPIC`
  - `Q_ASSIGNMENT` $\rightarrow$ `GET_ASSIGNMENT`
  - `Q_DEADLINE` $\rightarrow$ `GET_DEADLINE`
  - `Q_RESOURCE` $\rightarrow$ `GET_RESOURCE`
  - `Q_RULE` $\rightarrow$ `GET_RULE`
  - `Q_LO` $\rightarrow$ `GET_LO`
  - `Q_UNKNOWN` $\rightarrow$ `UNKNOWN`

### 8.2 Phương pháp trích xuất và chuẩn hóa Entity (`semantic_frame.py`, `semantic.py`, `knowledge_base.py`)
Quá trình trích xuất thực thể kết hợp **3 cơ chế bổ trợ lẫn nhau**:
1. **Duyệt nút thực thể trên cây cú pháp (`_first_node` & `_node_text`):**
   Duyệt theo thứ tự ưu tiên qua 13 loại nhãn nút thực thể trên cây cú pháp:
   * `COURSE_REF` $\rightarrow$ chuẩn hóa về `"CO3085"`.
   * `EXAM_REF` $\rightarrow$ kiểm tra bề mặt (`"midterm"` / `"giữa kỳ"` $\rightarrow$ `"MIDTERM"`, ngược lại $\rightarrow$ `"FINAL"`).
   * `WEEK_REF` $\rightarrow$ trích số nguyên bằng regex `\d+` và định dạng thành `"WEEK_{:02d}"` (`WEEK_01` .. `WEEK_15`).
   * `CHAPTER_REF` $\rightarrow$ trích số nguyên và định dạng thành `"CH{:02d}"` (`CH01` .. `CH12`).
   * `LO_REF` $\rightarrow$ trích chuỗi `\d+\.\d+` và định dạng thành `"LO2.5"`, `"LO3.3"`, `"LO4.2"`.
   * `ASSIGNMENT_REF`, `DEADLINE_REF` $\rightarrow$ chuẩn hóa về `"BTL01"`.
   * `RULE_OBJECT`, `RULE_REF` $\rightarrow$ phân biệt `"LLM_RAG"` và `"AI_USAGE"`.
   * `RESOURCE_TOPIC` $\rightarrow$ phân biệt `"PYTHON_NLP"`, `"STATISTICAL_NLP"`, `"MACHINE_TRANSLATION"`.
   * `TOPIC_REF`, `PART_REF`, `RESOURCE_REF` $\rightarrow$ tra cứu trong từ điển đồng nghĩa `aliases`.
2. **Đối sánh từ điển thực thể và đồng nghĩa (`load_entity_aliases` từ `data/entities.txt`):**
   * File `data/entities.txt` định nghĩa các cặp `alias = canonical_value` theo từng nhóm (`course`, `chapters`, `topics`, `learning_outcomes`, `assignments`, `policies`).
   * Hàm `load_entity_aliases()` chuẩn hóa các tên gọi tiếng Việt và tiếng Anh về cùng một `entity_id` (ví dụ: `"văn phạm phi ngữ cảnh"`, `"context-free grammar"`, `"CFG"` $\rightarrow$ `CFG`; `"gán nhãn từ loại"`, `"part-of-speech tagging"`, `"POS tagging"` $\rightarrow$ `POS_TAGGING`) và **sắp xếp giảm dần theo độ dài chuỗi** (`key=lambda item: len(item[0]), reverse=True`) để cụm từ cụ thể dài hơn luôn được ưu tiên khớp trước.
3. **Bắt thực thể số học theo mẫu cấu trúc (Structure-aware Regex Fallback):**
   * Quét bổ sung trên câu đã chuẩn hóa với biểu thức `\b(?:tuan|week)\s+(?:thu\s+)?0?(\d{1,2})\b` và `\b(?:chuong|chapter)\s+0?(\d{1,2})\b` để đảm bảo bắt đủ mọi chương/tuần ngay cả trong câu so sánh có nhiều chương (`"Chương 2 và Chương 3 khác nhau như thế nào?"` $\rightarrow$ `["CH02", "CH03"]`).
4. **Phân loại khe ngữ nghĩa (`entity_category` trong `semantic.py`):**
   * Gán từng `entity_id` vào đúng slot ngữ nghĩa (`chapter`, `week`, `part`, `assignment`, `course`, `lo`, `exam`, `rule`, `topic`) trong dictionary `entities` để phục vụ cập nhật `DialogueState`.

---

## 9. Mô tả Knowledge Base và cách truy vấn

### 9.1 Cơ sở lý thuyết: Biểu diễn tri thức và Truy xuất thông tin
* Trong bài giảng **Chương 1 (Information Retrieval & Information Extraction)** và **Chương 11 (Knowledge Representation and Reasoning – Procedural Semantics and Question Answering)**, một hệ thống hỏi đáp miền hẹp cần tổ chức cơ sở tri thức sao cho vừa bảo toàn ngữ cảnh văn bản tự nhiên, vừa gắn kết với các ký hiệu thực thể chuẩn hóa để thực hiện truy vấn chính xác.

### 9.2 Cấu trúc Knowledge Base và Phân đoạn dữ liệu (`knowledge_base.py`)
Knowledge Base được nạp từ 7 tập tin văn bản cộng 1 tập tin thực thể trong `python/data/` (nguồn dữ liệu chuẩn của môn học CO3085 do giảng viên cung cấp):

| file | Loại | Quy tắc phân đoạn | Nội dung chính |
|---|---|---|---|
| `course_info.txt` | `course_info` | Tách theo đoạn trống `\n\s*\n` | Mã môn `CO3085`, 3 tín chỉ, phân bổ giờ, mô tả 3 phần, chuẩn đầu ra `L.O.`. |
| `schedule.txt` | `schedule` | Tách theo tiêu đề `(?=^WEEK\s+\d+\b)` | Kế hoạch giảng dạy chi tiết từ `WEEK 01` đến `WEEK 15`, lịch thi Midterm/Final. |
| `topics.txt` | `topics` | Tách theo tiêu đề `(?=^CHAPTER\s+\d+\s*:)` | Nội dung 12 chương (`CHAPTER 1` – `CHAPTER 12`): `part`, `keywords`, `summary`, `subtopics`. |
| `assignments.txt` | `assignments` | Tách theo `(?=^(?:PART\s+[IVX]+\s*-|LLM/RAG EXTENSION))` | Yêu cầu Phần I, II, III, IV của bài tập lớn, các file output cần nộp, lưu ý deadline. |
| `regulations.txt` | `regulations` | Tách theo tên trường quy định (`attendance:`, `AI_usage:`, `LLM_RAG:`, ...) | Quy định điểm danh, trung thực học thuật, sử dụng AI, LLM/RAG, thư viện ngoài. |
| `resources.txt` | `resources` | Tách theo `(?=^(?:TEXTBOOK\s+\d+|REFERENCE\s+\d+|study_note\s*:))` | 3 giáo trình chính (Allen 1995; Jurafsky & Martin; Manning & Schütze) và 5 tài liệu tham khảo. |
| `faq.txt` | `faq` | Tách theo câu hỏi `(?=^Q\d+\s*:)` | Các cặp câu hỏi – trả lời thường gặp của môn học (`Q1` – `Q15`). |

Mỗi đoạn sau khi tách được đóng gói thành một dictionary **chunk**:
```python
{
    "chunk_id": "topics-chapter-6-ambiguity-resolution-statistical-methods-5",
    "text": "CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS ...",
    "source": "topics.txt",
    "section": "CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS",
    "entity_ids": ["CH06", "SYNTACTIC_PROCESSING", "POS_TAGGING", "PCFG"],
    "kind": "topics",
}
```

### 9.3 Cơ chế truy vấn lai (Hybrid Retrieval: TF-IDF + Semantic Frame trong `retriever.py`)
Khi nhận câu hỏi `question`, `Retriever.retrieve()` thực hiện quy trình 4 bước:
1. **Xây dựng chỉ mục TF-IDF có làm giàu thực thể (`__init__`):**
   * Mỗi chunk trong corpus được nối thêm chuỗi token đặc trưng `entityid<id>` (ví dụ `entityidch06`, `entityidpcfg`) lặp lại 2 lần nhằm tăng trọng số khớp thực thể chính xác.
   * Khởi tạo `TfidfVectorizer(preprocessor=normalize, lowercase=True, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)` với thang đo tần suất dưới tuyến tính $1 + \log(\text{tf})$.
2. **Mở rộng câu truy vấn (`expand_query`):**
   * Đối sánh các alias trong câu hỏi để chèn thêm từ khóa chuẩn tiếng Anh/Việt (`canonical`) và token `entityid<id>`.
   * Dựa trên `intent` của `SemanticFrame`, chèn thêm bộ từ khóa định hướng miền (`intent_terms`), ví dụ:
     - `GET_SCHEDULE` $\rightarrow$ `"schedule week weekly plan lesson topics"`
     - `GET_TOPIC` $\rightarrow$ `"chapter topic summary subtopics"`
     - `GET_RESOURCE` $\rightarrow$ `"resources textbook reference books"`
3. **Tính điểm kết hợp (Hybrid Scoring):**
   * **Điểm từ vựng ($\text{LexicalScore}$):** Độ tương đồng Cosine giữa vector TF-IDF của truy vấn mở rộng và ma trận chunk:
     $$\text{LexicalScore}(q, d) = \cos(\mathbf{v}_{\text{TF-IDF}}(q_{\text{expanded}}), \mathbf{v}_{\text{TF-IDF}}(d))$$
   * **Điểm ngữ nghĩa ($\text{SemanticScore}$ – hàm `_semantic_scores`):**
     Gọi $I_q$ là intent của câu hỏi, $E_q$ là tập thực thể của câu hỏi, $I_d$ và $E_d$ là tập intent và thực thể của chunk $d$:
     - Nếu $E_q = \emptyset$ (câu hỏi tổng quát không nêu thực thể cụ thể):
       $$\text{SemanticScore}(q, d) = \begin{cases} 1.0 & \text{nếu } I_q \in I_d \text{ và } \text{kind}(d) \neq \text{faq} \\ 0.1 & \text{nếu } I_q \in I_d \text{ và } \text{kind}(d) = \text{faq} \\ 0.0 & \text{ngược lại} \end{cases}$$
     - Nếu $E_q \neq \emptyset$ (câu hỏi có thực thể cụ thể):
       $$\text{SemanticScore}(q, d) = \text{IntentPrior}(q, d) + 0.8 \times \frac{|E_q \cap E_d|}{|E_q|}$$
       với $\text{IntentPrior}(q, d) = 0.2$ nếu $I_q \in I_d$ (hoặc $0.05$ nếu là chunk `faq`).
   * **Điểm tổng hợp ($\text{CombinedScore}$):** Với trọng số mặc định `lexical_weight = 0.75` và `semantic_weight = 0.25`:
     $$\text{CombinedScore}(q, d) = 0.75 \times \text{LexicalScore}(q, d) + 0.25 \times \text{SemanticScore}(q, d)$$
     *(Nếu câu hỏi `UNPARSED` không có intent hợp lệ, hệ thống tự động dùng $\text{CombinedScore}(q, d) = \text{LexicalScore}(q, d)$).*
4. **Lọc theo ngưỡng và Trả kết quả:**
   * Nếu `intent == "UNKNOWN"`, danh sách kết quả được xóa rỗng ngay lập tức (`selected = []`).
   * Ngược lại, sắp xếp giảm dần theo `CombinedScore` và lấy tối đa `top_k = 3` chunk thỏa mãn $\text{CombinedScore} \ge \text{min\_score} = 0.08$.
   * Trả về `retrieval_status = "FOUND"` nếu có ít nhất 1 chunk được chọn, hoặc `"NO_RELEVANT_CONTEXT"` nếu rỗng.

---

## 10. Mô tả lựa chọn LLM/RAG

Tài liệu module chi tiết, gồm label audit, multi-intent, API và answer-generation telemetry: [python/nlp/app/README.md](nlp/app/README.md).

Nhóm lựa chọn **Lựa chọn 2: Mở rộng bằng LLM/RAG** đồng thời hoàn thiện đầy đủ **Xử lý ngữ cảnh hội thoại cục bộ**. Đúng theo yêu cầu của đề bài, phần mở rộng LLM/RAG **không thay thế** các thành phần NLP cốt lõi mà sử dụng trực tiếp kết quả của Grammar, Earley Parser và Semantic Frame từ Phần I.

### 10.1 Mô hình sử dụng và Cấu hình (`llm_client.py`)
Hệ thống định nghĩa giao diện trừu tượng `LLMClient` với 2 cài đặt cụ thể:
1. **`MockLLMClient` (`model_name = "mock-grounded-extractive-v1"` – Mặc định Offline):**
   * Hoạt động hoàn toàn offline, không cần kết nối mạng hay API key, hỗ trợ tái lập kết quả khi dữ liệu và cấu hình không thay đổi.
   * Cơ chế: Phân tích câu hỏi và các chunk trong prompt, chọn lọc các câu/mục chứa thông tin khớp trực tiếp với câu hỏi và đính kèm nguyên văn nhãn trích dẫn `[Nguồn: <source> / <section>]`.
2. **`OpenAICompatibleLLMClient` (Chế độ gọi LLM thật qua HTTP API):**
   * Kết nối tới mọi endpoint tương thích chuẩn OpenAI Chat Completions (`/v1/chat/completions`), mặc định dùng model `gpt-4o-mini` với `temperature = 0.0` để tối đa hóa tính nhất quán.
   * Cấu hình hoàn toàn qua biến môi trường (không lưu khóa bí mật trong mã nguồn):

| Biến môi trường | Tham số CLI | Giá trị mặc định | Mô tả |
|---|---|---|---|
| `LLM_PROVIDER` | `--provider` | `mock` | Chọn `mock`, `openai`, hoặc `auto` (tự động dùng `openai` nếu có `OPENAI_API_KEY`). |
| `LLM_MODEL` | `--model` | `gpt-4o-mini` | Tên model LLM sử dụng khi gọi API. |
| `OPENAI_API_KEY` | *(không)* | *(trống)* | Khóa API (không bao giờ được in ra log hay file output). |
| `OPENAI_BASE_URL` | *(không)* | `https://api.openai.com/v1` | Địa chỉ máy chủ API tương thích OpenAI. |
| `LLM_TIMEOUT` | *(không)* | `30` | Thời gian chờ tối đa (giây) cho mỗi lượt gọi API. |

   * **Cơ chế Fallback an toàn:** Nếu người dùng chọn `openai` nhưng quên đặt `OPENAI_API_KEY` hoặc gặp lỗi kết nối mạng, hệ thống tự động chuyển về `MockLLMClient` và ghi rõ vào header của file output: `mock-grounded-extractive-v1 [FALLBACK: ..., requested gpt-4o-mini]`.

### 10.2 Cách xây dựng Prompt và Context (`prompt_builder.py`)
Hàm `build_prompt(question, chunks, semantic=None, dialogue_state=None)` đóng gói thông tin thành cấu trúc chặt chẽ gồm 4 khối:
1. **Chỉ dẫn hệ thống (System Instructions):**
   * Chỉ được trả lời dựa trên thông tin nằm trong khối `CONTEXT` bên dưới.
   * Tuyệt đối không suy diễn hoặc bịa đặt ngày hạn nộp, thời gian, phòng học, tên giảng viên hay quy định không có trong `CONTEXT`.
   * Mọi câu khẳng định phải gắn kèm nhãn trích dẫn sao chép nguyên văn theo đúng định dạng `[Nguồn: <source> / <section>]`.
   * Nếu các đoạn `CONTEXT` không chứa đủ thông tin để trả lời câu hỏi, bắt buộc trả về đúng một chuỗi duy nhất: `INSUFFICIENT_CONTEXT`.
2. **Gợi ý ngữ nghĩa từ Phần I (Semantic Hint):**
   * Cung cấp `intent` và `entities` do Earley Parser trích xuất để định hướng trọng tâm câu trả lời, kèm ghi chú rõ: *(Gợi ý cấu trúc câu hỏi, không phải bằng chứng thay thế CONTEXT)*.
3. **Danh sách tài liệu truy xuất (Retrieved Context Blocks):**
   * Liệt kê từng chunk theo định dạng:
     ```text
     --- CONTEXT 1 [Nguồn: topics.txt / CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS] ---
     <nội dung văn bản của chunk>
     ```
4. **Câu hỏi đã giải tham chiếu (`resolved_query`):** Đặt ở cuối prompt để mô hình tập trung trả lời.

### 10.3 Cơ chế Answer Guard 3 lớp kiểm tra grounding (`answer_guard.py`, `assistant.py`)
Để giảm nguy cơ sinh ảo (*hallucination*) của LLM, hệ thống áp dụng 3 lớp kiểm duyệt độc lập:
* **Lớp 1 – Tiền kiểm tra trước khi gọi LLM (`check_context`):**
  1. Nếu `retrieval_status != "FOUND"` hoặc danh sách `chunks` rỗng $\rightarrow$ trả về ngay `NO_RELEVANT_CONTEXT`.
  2. Nếu văn phạm Phần I phân loại câu hỏi vào `intent == "UNKNOWN"` $\rightarrow$ trả về ngay `NO_RELEVANT_CONTEXT`.
  3. Nếu câu hỏi không khớp được bất kỳ intent hay entity nào (`UNPARSED` và `entity_ids` rỗng), hàm `lexical_support(question, chunks)` loại bỏ danh sách từ hư tiếng Việt (`_STOPWORDS`: *"là"*, *"gì"*, *"hôm"*, *"nay"*, *"thế"*, *"nào"*, ...) và tính tỷ lệ từ nội dung của câu hỏi xuất hiện trong các chunk. Nếu không có từ nội dung nào xuất hiện (`lexical_support <= 0.0`) $\rightarrow$ chặn ngay với `NO_RELEVANT_CONTEXT`.
* **Lớp 2 – Kiểm tra tính đầy đủ của thông tin & Sentinel (`INSUFFICIENT_CONTEXT`):**
  1. Trước và sau LLM, hệ thống kiểm tra các câu hỏi về thuộc tính mà KB chỉ chứa bản mẫu chưa có giá trị thực tế (ví dụ: hỏi **ngày giờ hạn nộp/deadline** chính thức của bài tập lớn trong khi `assignments.txt` chỉ ghi *"Lưu ý: dữ liệu mẫu, cần thay bằng lịch chính thức"*, hoặc hỏi **phòng học/địa điểm học**, hoặc hỏi chương vượt quá Chương 12 như `CH13`).
  2. Nếu mô hình trả về chuỗi sentinel `INSUFFICIENT_CONTEXT` $\rightarrow$ hệ thống trả trạng thái `INSUFFICIENT_CONTEXT` với thông báo lịch sự, không gắn nguồn giả.
* **Lớp 3 – Hậu kiểm tra căn cứ và xác thực thực thể số/ngày (`check_answer`):**
  1. **Xác thực nguồn trích dẫn (`Citation Validation`):** Trích xuất mọi nhãn `[Nguồn: source / section]` bằng biểu thức `_CITATION` và đối chiếu với tập các cặp `(source, section)` thực sự đã đưa vào prompt (`allowed`). Mọi trích dẫn bịa đặt (`invalid_citations`) bị xóa bỏ ngay lập tức; nếu không còn trích dẫn hợp lệ nào hoặc có trích dẫn bịa, trạng thái bị hạ xuống `NEEDS_REVIEW`.
  2. **Xác thực thông tin định lượng (`Fact Verification` – hàm `unsupported_facts`):**
     * Sử dụng biểu thức chính quy `_FACT` quét toàn bộ **ngày tháng** (`\d{1,2}[/-]\d{1,2}...`), **giờ** (`\d{1,2}[:h]\d{2}`), **địa chỉ email**, **URL** và **con số** (`\d+(?:[.,]\d+)?`) trong câu trả lời của mô hình (sau khi đã tách nhãn `[Nguồn: ...]`).
     * Chuẩn hóa các con số qua `_norm_fact` (bỏ số `0` ở đầu để `"tuần 5"` khớp hợp lệ với `"WEEK 05"`) và kiểm tra xem từng giá trị có xuất hiện trong văn bản gốc của các chunk đã gửi hay không.
     * Nếu phát hiện bất kỳ con số, ngày tháng hay URL nào do mô hình tự bịa ra (`unsupported_facts` khác rỗng), hệ thống **tịch thu câu trả lời**, gán trạng thái `NEEDS_REVIEW`, lưu câu gốc vào `raw_answer` để kiểm toán và trả thông báo an toàn cho người dùng:
       *"Câu trả lời do mô hình sinh ra chứa thông tin không có trong tài liệu truy xuất (...), nên hệ thống không đưa ra để tránh sai lệch."*

### 10.4 Xử lý ngữ cảnh hội thoại cục bộ – Phần III (`dialogue.py`)
Theo lý thuyết **Chương 7 (Discourse Situation)** và **Chương 12 (Discourse Context and Reference)**, ý nghĩa của một phát ngôn phụ thuộc vào trạng thái diễn ngôn hiện tại $D_{\text{initial}}$ và sau khi trả lời sẽ cập nhật thành trạng thái diễn ngôn mới $D_{\text{updated}}$.

1. **Cấu trúc trạng thái `DialogueState`:**
   ```python
   {
     "last_question": "Chương 6 nói về gì?",
     "last_intent": "GET_TOPIC",
     "last_entities": {"chapter": "CH06", "part": "PART_I", "topic": "POS_TAGGING"},
     "last_answer_entities": ["CH06", "POS_TAGGING", "PCFG"],
     "last_sources": [{"source": "topics.txt", "section": "CHAPTER 6: AMBIGUITY RESOLUTION - STATISTICAL METHODS"}]
   }
   ```
   Ngoài ra, `DialogueState` lưu thêm `_ambiguous_candidates` khi câu hỏi trước nhắc tới nhiều thực thể cùng loại (ví dụ: `{"chapter": ["CH02", "CH03"]}`).

2. **Sáu quy tắc giải tham chiếu cục bộ (`resolve_reference` trong `dialogue.py`):**

| Loại tham chiếu| Dấu hiệu nhận diện trong câu hỏi | Quy tắc biến đổi sang query | Ví dụ thực tế  |
|---|---|---|---|
| `PRONOUN_IT` | Đại từ *"nó"* làm chủ ngữ/bổ ngữ | Thay *"nó"* bằng tên hiển thị chuẩn (`ENTITY_DISPLAY`) của thực thể trung tâm ở lượt trước (`chapter` $\succ$ `topic` $\succ$ `week` $\succ$ `lo`). | Lượt trước: *"Chương 6 nói về gì?"* (`CH06`)<br>Câu hỏi: *"Nó có liên quan đến gán nhãn từ loại không?"*<br>$\rightarrow$ `"Chương 6 có liên quan đến gán nhãn từ loại không?"` |
| `THIS_CHAPTER` | Cụm chỉ xuất *"chương này"*, *"chương đó"* | Thay bằng `"Chương X"` lấy từ `last_entities["chapter"]`. | Lượt trước: `CH06`<br>Câu hỏi: *"Chương này học vào tuần nào?"*<br>$\rightarrow$ `"Chương 6 học vào tuần nào?"` |
| `NEXT_CHAPTER` | Câu hỏi rút gọn *"Còn chương sau?"*, *"Thế còn chương tiếp theo?"* | Tăng chỉ số chương hiện tại lên $X + 1$ (`CH06` $\rightarrow$ `CH07`) và kế thừa kiểu câu hỏi của lượt trước. Nếu $X + 1 > 12$ (`CH13`), đánh dấu vượt phạm vi KB. | Lượt trước: *"Chương 8 học vào tuần nào?"* (`CH08`)<br>Câu hỏi: *"Còn chương sau?"*<br>$\rightarrow$ `"Chương 9 học vào tuần nào?"` |
| `THIS_CONTENT` | Cụm chỉ xuất *"nội dung này"*, *"chủ đề này"* | Thay bằng thực thể chương/chủ đề/L.O. vừa thảo luận ở lượt trước. | Lượt trước: `CH09`<br>Câu hỏi: *"Nội dung này học vào tuần nào?"*<br>$\rightarrow$ `"Chương 9 học vào tuần nào?"` |
| `THIS_PART` | Cụm chỉ xuất *"phần này"*, *"phần đó"* (khi đang hỏi về bài tập lớn) | Thay bằng `"Phần I/II/III/IV"` hoặc `"bài tập lớn"` từ `last_entities`. | Lượt trước: *"Bài tập lớn có phần hội thoại không?"*<br>Câu hỏi: *"Phần này cần xuất file gì?"*<br>$\rightarrow$ `"bài tập lớn cần xuất file gì?"` |
| `RESOURCE_OF_THIS_PART` | *"Tài liệu của phần này/chương này ở đâu / là gì?"* | Kết hợp thực thể chính và các chủ đề trong câu trả lời lượt trước (`last_answer_entities`) thành truy vấn tài liệu đầy đủ. | Lượt trước: *"Tuần 2 học những nội dung gì?"* (`WEEK_02`, `CH02`)<br>Câu hỏi: *"Tài liệu của phần này ở đâu?"*<br>$\rightarrow$ `"Tài liệu tham khảo cho Tuần 2 (Phần I, Chương 2) là gì?"` |

3. **Xử lý thiếu tiền từ hoặc tiền từ mơ hồ (`NEED_CLARIFICATION`):**
   * **Thiếu tiền từ (Missing Antecedent):** Khi bắt đầu cuộc hội thoại mới hoặc vừa gõ lệnh `RESET` (`DialogueState.is_empty() == True`) mà người dùng hỏi ngay *"Nó học vào tuần nào?"* hoặc *"Còn chương sau?"*, hệ thống không đoán mò mà trả về `status = "NEED_CLARIFICATION"` yêu cầu người dùng nêu rõ chương/chủ đề muốn hỏi.
   * **Tiền từ mơ hồ (Ambiguous Antecedent):** Khi lượt trước người dùng hỏi so sánh 2 chương (*"Chương 2 và Chương 3 khác nhau như thế nào?"*), `_ambiguous_candidates["chapter"] = ["CH02", "CH03"]`. Ở lượt tiếp theo nếu người dùng hỏi *"Chương này học vào tuần nào?"*, hệ thống phát hiện sự nhập nhằng tham chiếu và trả về `status = "NEED_CLARIFICATION"`: *"Trong lượt trước có nhiều hơn một chương (Chương 2, Chương 3). Bạn muốn hỏi về Chương 2 hay Chương 3?"*.

---

## 10.5. Xử lý câu hỏi nhiều intent – Grammar v5

Grammar v5 nhận diện câu hỏi ghép khi có thể phân hoạch chúng thành các clause độc lập đều được CFG hiện tại chấp nhận. Splitter không chia máy móc tại mọi từ “và”: các cụm thuật ngữ hoặc cấu trúc so sánh vẫn có thể được giữ nguyên thành một intent nếu grammar nhận diện đó là một cấu trúc duy nhất.

Luồng xử lý:

1. Splitter đề xuất các clause và kiểm tra từng clause bằng CFG.
2. Nếu chỉ có một clause hợp lệ, hệ thống giữ nguyên pipeline single-intent.
3. Nếu có nhiều clause độc lập, mỗi clause được parse và tạo Semantic Frame riêng.
4. Retriever và answer-generation pipeline chạy riêng cho từng clause.
5. Assistant hợp nhất câu trả lời, chunks, sources và entities thành kết quả chung.

**Lưu ý kiến trúc:** hiện tại đây là xử lý *per-clause*. Một câu hỏi có N intent có thể tạo N lần gọi `LLMClient.generate()` nếu các clause vượt qua pre-generation Answer Guard. Đây không phải thiết kế gom tất cả intent vào một prompt và gọi LLM đúng một lần. Guard có thể chặn một clause trước khi gọi model, vì vậy số lần gọi thực tế có thể nhỏ hơn số clause.

Hợp đồng kết quả single-intent vẫn giữ các trường legacy `intent`, `entities`, `entity_ids`, `parse_status`. Kết quả multi-intent bổ sung `intent: MULTI_INTENT`, `intents` và `clauses`. Chi tiết: [GRAMMAR_V5_MULTI_INTENT.md](GRAMMAR_V5_MULTI_INTENT.md).

## 10.6. Telemetry cho LLM generation

`LLMClient` lưu metadata cho từng lần gọi `generate()` trong `call_history`:

| Field | Ý nghĩa |
|---|---|
| `provider` | Provider được cấu hình |
| `requested_model` | Model được yêu cầu |
| `model_name` | Model thực tế hoặc tên mock/fallback |
| `attempted_api` | Có thực sự bắt đầu thử gọi API hay không |
| `used_api` | API trả về generation thành công |
| `fallback` | Lần gọi chuyển sang Mock |
| `error` | Loại lỗi, không chứa nội dung request |
| `elapsed_seconds` | Thời gian của lần gọi generation |

`generation_count` đếm số lần gọi `generate()`; `last_call` trả metadata của lần gần nhất. Telemetry không lưu prompt, câu hỏi, retrieved context hoặc API key. Một câu bị Answer Guard chặn trước bước generation sẽ không tạo telemetry record.

Đối với đánh giá model thật, chỉ `used_api = true` mới được tính là API generation thành công. Chỉ chọn provider `openai` không đủ để chứng minh API đã được sử dụng, vì request có thể thất bại và fallback về Mock.

## 11. Mô tả cách đánh giá



### 11.1 Parser, Semantic, Retrieval và deterministic end-to-end

Chạy từ thư mục `python/`:

~~~bash
python -m nlp.app.evaluation
python -m nlp.app.evaluation --samples nlp/app/output/samples_reviewed.jsonl --max-samples 100
~~~

Evaluator đo parse status/tree, intent/entity, retrieval Hit@k, answerability/refusal, citation và Answer Guard. Retrieval aggregate dùng query-only, không đưa gold intent/entity vào retriever. Mặc định pipeline deterministic dùng Mock để có thể tái lập.

### 11.2 Đánh giá câu trả lời được sinh ra

`answer_evaluation.py` chạy dữ liệu JSONL qua CourseAssistant và lưu provenance của các lần generation:

~~~bash
# Offline/mock evaluation
python -m nlp.app.answer_evaluation --provider mock

# Gọi API thật (có thể fallback nếu request lỗi)
export OPENAI_API_KEY="YOUR_API_KEY"
python -m nlp.app.answer_evaluation --provider openai --model gpt-4o-mini --max-samples 20

# Bắt buộc tất cả hàng được đánh giá phải có API generation thành công, không fallback
python -m nlp.app.answer_evaluation --provider openai --require-real-model --max-samples 20
~~~

Mặc định input là `nlp/app/input/answer_eval_smoke.jsonl`; có thể thay bằng `--samples PATH`. Output nằm trong `nlp/app/output/answer_evaluation/`.

Các field quan trọng gồm `model_execution` (`real_api`, `mock`, `fallback`, `mixed` hoặc `unknown`), `model_call_count`, `model_calls`, `model_fallback`, `model_error`, `generation_elapsed_seconds` và `elapsed_seconds`.

Với câu hỏi multi-intent, evaluation lấy các record mới phát sinh trong `call_history` trong suốt một lần `assistant.answer()`. Do mỗi clause chạy riêng, một hàng đánh giá có thể có nhiều generation calls và trạng thái `mixed` nếu có cả API thành công lẫn fallback.

### 11.3 Unified evaluation

`unified_evaluation.py` chạy parser/semantic/retrieval evaluation và answer-generation evaluation trong cùng một report:

~~~bash
python -m nlp.app.unified_evaluation --provider mock --max-samples 10

# Chỉ dùng khi muốn xác minh generation bằng API thật
python -m nlp.app.unified_evaluation \
  --provider openai \
  --require-real-model \
  --max-samples 5
~~~

Tùy chọn chính: `--nlp-samples`, `--answer-samples`, `--data-dir`, `--grammar`, `--provider`, `--model`, `--judge`, `--judge-model`, `--max-samples`, `--require-real-model`, `--output-dir`.

Mặc định NLP samples là `nlp/app/output/samples_reviewed.jsonl`, answer samples là `nlp/app/input/answer_eval_smoke.jsonl`. Output được ghi tại `nlp/app/output/unified_evaluation/unified_evaluation.json` và `unified_evaluation.md`.

`--require-real-model` yêu cầu mỗi hàng được đánh giá phải có ít nhất một generation call và tất cả các call trong hàng đó phải là API thành công, không fallback. Hàng bị pre-generation guard chặn nên không có generation call sẽ làm chế độ này thất bại có chủ đích; điều đó không đồng nghĩa telemetry bị mất.

### 11.4 Model-as-judge và cách đọc metric

Có thể bật judge riêng bằng `--judge --judge-model MODEL_ID`. Judge dùng API riêng; nên chọn model khác model sinh câu trả lời nếu mục tiêu là giảm thiên lệch tự đánh giá. Judge score là đánh giá phụ trợ, không phải ground truth. Correctness/completeness chỉ được chấm khi có reference answer/points phù hợp và đã được kiểm chứng; output mock/fallback/mixed không được dùng để chấm chất lượng model thật.

### 11.5 Nguyên tắc báo cáo

- Chỉ dùng nhãn đã review độc lập cho headline metrics; phân biệt human-review với auto labels.
- Grammar-generated samples hữu ích cho regression/coverage, nhưng không chứng minh độ chính xác trên câu hỏi tự nhiên ngoài đời.
- Report rõ provider, model ID, số API calls thành công, fallback, số mẫu, dataset và cấu hình.
- Không gộp kết quả Mock với kết quả model thật.
- Multi-intent evaluation phải báo cáo số clause và số generation calls; không diễn giải per-clause calls như một lần gọi LLM duy nhất.
- Citation hợp lệ chứng minh citation trỏ tới context đã gửi, không tự nó chứng minh entailment ngữ nghĩa đầy đủ.

---

## 12. Các hạn chế của hệ thống

1. **Tính cứng nhắc của văn phạm CFG biên soạn thủ công:**
   * Văn phạm phi ngữ cảnh ở Phần I được thiết kế để nhận diện các mẫu câu đã định nghĩa, nhưng nếu người dùng sử dụng câu khẩu ngữ rút gọn, đảo trật tự từ tự do, viết tắt hoặc sai dấu tiếng Việt (ngoài danh sách alias), Earley Parser sẽ trả về `()` (`UNPARSED`).
2. **Giới hạn biểu diễn từ vựng của TF-IDF:**
   * Tầng truy xuất dựa trên TF-IDF kết hợp từ điển đồng nghĩa tĩnh (`data/entities.txt`). Nếu người dùng diễn đạt bằng các cụm từ đồng nghĩa hoặc cách mô tả khái niệm hoàn toàn mới chưa có trong `entities.txt` (không chia sẻ từ vựng bề mặt với tài liệu), điểm `LexicalScore` sẽ thấp. Có thể khắc phục trong tương lai bằng cách kết hợp thêm Dense Embedding (Semantic Search đa ngôn ngữ).
3. **Sự khác biệt giữa `MockLLMClient` (Offline) và LLM thực thụ (Online):**
   * `MockLLMClient` cho phép hệ thống chạy offline, không cần API key và hỗ trợ tái lập kết quả trong cùng cấu hình, nhưng câu trả lời có tính chất trích xuất nguyên đoạn (*extractive*) nên đôi khi dài hơn mức cần thiết và không tự suy luận để từ chối các câu ngoài miền có trùng từ vựng ngẫu nhiên như khi chạy với `OpenAICompatibleLLMClient`.
4. **Phạm vi của bộ kiểm tra thực thể trong `answer_guard.py`:**
   * Hậu kiểm tra `unsupported_facts` kiểm tra con số, ngày tháng, giờ giấc, email, URL và trích dẫn nguồn bằng luật; đây không phải bộ phát hiện hallucination hoàn chỉnh và có thể có false positive/false negative. Tuy nhiên, nếu một mô hình ngôn ngữ giữ nguyên các con số trong nguồn nhưng đảo ngược ý nghĩa mệnh đề bằng chữ (ví dụ đổi *"bắt buộc"* thành *"không bắt buộc"*), bộ lọc biểu thức chính quy chưa thể tự phát hiện mà cần tới mô hình Natural Language Inference (NLI).
5. **Giới hạn của quản lý ngữ cảnh hội thoại cục bộ 1 bước (`DialogueState`):**
   * Mô-đun `dialogue.py` được thiết kế tối ưu cho các tham chiếu cục bộ liền kề (1-step anaphora & ellipsis resolution) theo đúng yêu cầu Phần III của đề bài. Hệ thống chưa hỗ trợ xếp chồng nhiều chủ đề hội thoại (dialogue stack) hoặc quay lại tham chiếu một thực thể đã nhắc tới cách đó 3–4 lượt hội thoại trước.


6. **Giới hạn của Grammar v5 multi-intent:** splitter phụ thuộc vào CFG hiện có và chỉ tách khi tìm được các clause độc lập đều parse hợp lệ. Câu ghép khẩu ngữ, thiếu dấu câu hoặc có cấu trúc vượt grammar có thể vẫn bị xử lý như single-intent hoặc UNPARSED. Mỗi clause được retrieval/generation riêng; số lần gọi LLM có thể tăng theo số clause và không phải một request duy nhất cho toàn câu.

7. **Giới hạn của real-model evaluation:** `--require-real-model` chỉ xác minh các generation calls đã diễn ra và thành công. Nếu Answer Guard chặn câu hỏi trước generation, không có API call để xác minh và hàng đó không thỏa chế độ real-model bắt buộc. Đây là hành vi guard, không phải lỗi telemetry. Model-as-judge cũng là tín hiệu tham khảo, không thay thế human-reviewed reference.
>>>>>>> Stashed changes
