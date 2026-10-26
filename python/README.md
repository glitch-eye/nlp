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
python -m hcmut.iaslab.nlp.app.run_parser --input input/sentences.txt --output output/parse-results.txt --grammar output/grammar.txt
```
Đối với mỗi câu:
- Hợp lệ: xuất cây cú pháp dạng ngoặc `(S ...)`.
- Không hợp lệ / ngoài văn phạm: xuất `()`.

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

---

## 7. Đánh giá và hạn chế của hệ thống
* **Ưu điểm:**
  - Tốc độ xử lý cực nhanh, không phụ thuộc kết nối API hay tài nguyên tính toán lớn.
  - Phân tích cú pháp chính xác, cây cú pháp rõ ràng, dễ dàng kiểm soát lỗi.
  - Bộ sinh câu tạo ra 10.000 câu tự nhiên, thuần Việt, đúng ngữ cảnh môn học.
* **Hạn chế:**
  - Phụ thuộc vào tập luật văn phạm; các câu hỏi có cách diễn đạt quá sai lệch hoặc nhiều tiếng lóng sẽ bị từ chối (`()`).
  - Cần mở rộng thêm từ đồng nghĩa và từ vựng phong phú hơn khi mở rộng sang các môn học khác.
