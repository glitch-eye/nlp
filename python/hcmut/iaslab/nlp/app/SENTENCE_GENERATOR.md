# Sentence generator (Phần 3.2)

Thiết kế tách biệt ba phần:

- `grammar_file.py`: đọc luật CFG từ `output/grammar.txt`.
- `course_grammar.py`: adapter cho generator; tiếng Việt dùng trực tiếp CFG trên,
  tiếng Anh dùng production rules trong module.
- `course_lexicon.py`: `CourseLexicon` đọc các rule terminal của CFG thành
  lexicon, bên cạnh danh sách terminal dựng sẵn cho tiếng Anh.
- `sentence_generator.py`: thuật toán mở rộng non-terminal theo luật sinh,
  khử câu trùng, giới hạn tối đa 10.000 câu và ghi UTF-8, mỗi dòng một câu.

Chạy sinh câu tiếng Việt:

```python
from hcmut.iaslab.nlp.app.course_grammar import CourseGrammar
from hcmut.iaslab.nlp.app.sentence_generator import generate_to_file

generate_to_file(CourseGrammar("vi"), "output/samples.txt")
```

Để sinh tiếng Anh, đổi `"vi"` thành `"en"`. Nếu muốn cả hai ngôn ngữ trong
cùng file, chạy generator cho từng grammar rồi nối danh sách và loại trùng.

Các luật quy định cấu trúc câu; lexicon cung cấp terminal để lấp các vị trí
trong luật. `POS_BY_CATEGORY` ghi loại từ/cụm từ của từng nhóm terminal, ví dụ
`TOPIC_NP` là cụm danh từ, `V_TOPIC` là cụm động từ và `PREPOSITION` là giới từ.
Các nhóm terminal nhiều từ được giữ như một đơn vị để tránh ghép từ tùy ý.

Nội dung hiện tại là khung ban đầu; cần bổ sung/đối chiếu terminal và
production với `entities.txt`, `topics.txt`, dữ liệu lịch và grammar chính thức
của nhóm. Grammar ở đây hiện là grammar Course Assistant minh họa; nếu module
Phần I đã có production rules riêng, adapter của module đó nên thay
`CourseGrammar` khi gọi `generate_to_file`.

Có thể truyền đường dẫn grammar khác khi khởi tạo, ví dụ
`CourseGrammar("vi", "data/grammar.txt")`. Các terminal trong file phải được
đặt trong dấu ngoặc kép; nonterminal dùng dạng `<SYMBOL> ::= ...`.
