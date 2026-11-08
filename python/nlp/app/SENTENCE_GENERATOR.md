# Sentence generator (Phần 3.2)

`grammar_file.py` đọc production từ `output/grammar.txt`. `course_grammar.py` là adapter nhỏ để đưa production đó vào `SentenceGenerator`; grammar file là nguồn duy nhất cho các câu tiếng Việt. `sentence_generator.py` mở rộng production, bỏ câu trùng và giới hạn tối đa 10.000 câu.

Chạy trực tiếp từ thư mục `sample-code-v1.0/python/`:

```bash
python3 hcmut/iaslab/nlp/app/utils.py
```

Mặc định ghi ra `hcmut/iaslab/nlp/app/output/samples.txt`, độc lập với thư mục hiện hành. Có thể truyền `--output` hoặc `--max-sentences`.

## Bộ mẫu có nhãn

Để sinh câu, parse lại, gán intent/entity và đối chiếu chunk kỳ vọng:

```bash
PYTHONPATH=hcmut python3 -m iaslab.nlp.app.build_sample_dataset
```

Lệnh tạo `output/samples.txt` và `output/samples_labeled.jsonl`. Mỗi nhánh gốc `Q_*` trong grammar tương ứng với một intent. Các nhãn tự động cần được rà trước khi dùng làm gold evaluation set. Chi tiết schema và giới hạn nằm trong `SAMPLE-DATASET.md`.

`output/parse-results.txt` mặc định tương ứng với `input/sentences.txt`. Chạy parser từ `sample-code-v1.0/python/` bằng:

```bash
PYTHONPATH=hcmut python3 -m iaslab.nlp.app.run_parser
```

## Phạm vi grammar

Grammar chứa các dạng câu hỏi tiếng Việt của Course Assistant. Nó được viết gọn theo intent và entity trong sample knowledge base để tránh sinh tổ hợp vô nghĩa. Tập câu generated hữu hạn này dùng cho kiểm tra parser/semantic mapping; cần thêm câu viết tay để đánh giá độ tự nhiên và độ bao phủ.
