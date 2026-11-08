# Bộ câu hỏi synthetic có nhãn

Grammar cũ sinh ra nhiều cụm/câu không có ý nghĩa cho Course Assistant do cho phép các nhánh câu tổng quát và đệ quy kết hợp tự do. `output/grammar.txt` hiện giới hạn ở các dạng hỏi có căn cứ trong sample knowledge base, với nhánh gốc riêng cho từng intent. Tên nhánh `Q_*` được dùng để suy ra intent.

## Tạo lại bộ mẫu

Từ `sample-code-v1.0/python/` trong repo:

```bash
PYTHONPATH=hcmut python3 -m iaslab.nlp.app.build_sample_dataset
```

Lệnh sinh lại:

- `app/output/samples.txt`: mỗi câu một dòng.
- `app/output/samples_labeled.jsonl`: một JSON record cho mỗi câu, gồm query, parse status/tree, intent, entities, chunk IDs kỳ vọng, mức đủ bằng chứng và trạng thái rà nhãn.

Thay đường dẫn/giới hạn bằng `--grammar`, `--data-dir`, `--output-jsonl`, `--output-text` và `--max-sentences`.

## Gán nhãn

Builder sinh câu từ grammar, parse lại từng câu bằng `CourseParser`, lấy intent từ nhánh `Q_*`, rồi chuẩn hóa tuần/chương/LO/topic/assignment/resource/rule thành entity IDs. `expected_chunk_ids` được đối chiếu với chunk metadata của knowledge base. Các nhãn tự động đều mang `AUTO_LABEL_REQUIRES_REVIEW`; nhóm cần xem lại các câu, intent, entity và evidence trước khi dùng làm gold evaluation set.

Deadline được gán `INSUFFICIENT_CONTEXT`: dataset chỉ nói đây là thông tin mẫu, không nêu hạn nộp cụ thể. Những câu hỏi ngoài miền không được sinh từ grammar này; dùng Q026–Q028 trong `sample_queries.txt` làm negative/out-of-scope cases riêng.

## Phạm vi và giới hạn

Bộ hiện tại có 1.173 câu theo 8 intent: `GET_COURSE_INFO`, `GET_SCHEDULE`, `GET_TOPIC`, `GET_ASSIGNMENT`, `GET_DEADLINE`, `GET_RESOURCE`, `GET_RULE`, `GET_LO`. Toàn bộ câu được grammar sinh ra và parse thành công, nhưng điều đó chỉ xác nhận tương thích giữa grammar và parser. Nó không đánh giá độ tự nhiên của câu hoặc độ bao phủ câu hỏi người dùng. Hãy bổ sung câu viết tay, rà nhãn, và giữ một phần riêng làm tập kiểm tra.
