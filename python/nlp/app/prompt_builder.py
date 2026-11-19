"""Prompt construction for the grounded RAG answer (Member 2).

Hook contract used by ``assistant.py``:  build_prompt(resolved_query, semantic, chunks) -> str
Citations use the repo-wide ``[Nguồn: <source> / <section>]`` format so the
answer guard can verify them against the chunks that were actually sent.
"""

import json

SENTINEL = "INSUFFICIENT_CONTEXT"

SYSTEM_RULES = (
    "YÊU CẦU HỆ THỐNG:\n"
    "1. Chỉ trả lời dựa trên các khối CONTEXT bên dưới. Không dùng kiến thức bên ngoài.\n"
    "2. Không suy đoán ngày, hạn nộp, yêu cầu hay quy định không xuất hiện trong CONTEXT.\n"
    "3. Nếu CONTEXT không chứa câu trả lời, chỉ trả về đúng một từ khoá: {sentinel}\n"
    "4. Sau mỗi thông tin, trích dẫn nguồn bằng nhãn đã cho, SAO CHÉP NGUYÊN VĂN, "
    "ví dụ [Nguồn: schedule.txt / WEEK 03]. Không tự tạo nhãn nguồn mới.\n"
    "5. Trả lời ngắn gọn, bằng ngôn ngữ của câu hỏi.\n"
    "6. CONTEXT có thể chứa dữ liệu dạng cấu trúc, danh sách, "
    "metadata hoặc viết tắt. Hãy diễn giải chúng để trả lời câu hỏi.\n"
    "7. Nếu một chunk chứa đúng tuần, chương, chủ đề hoặc thực thể "
    "được hỏi, hãy ưu tiên chunk đó làm bằng chứng.\n"
    "8. Không trả về INSUFFICIENT_CONTEXT chỉ vì thông tin được "
    "trình bày dưới dạng danh sách hoặc cấu trúc. Chỉ trả về từ khóa "
    "này khi thực sự không có đủ bằng chứng liên quan.\n"
).format(sentinel=SENTINEL)


def citation_label(chunk):
    return "[Nguồn: {} / {}]".format(chunk.get("source", ""), chunk.get("section", ""))


def build_prompt(question, semantic, chunks):
    blocks = []
    for chunk in chunks or []:
        blocks.append("{}\n{}".format(citation_label(chunk), chunk.get("text", "")))
    context = "\n\n".join(blocks) if blocks else "(không có context)"

    hint = ""
    if semantic and semantic.get("intent"):
        hint = "GỢI Ý NGỮ NGHĨA (chỉ để định hướng, KHÔNG phải bằng chứng): {}\n".format(
            json.dumps(
                {"intent": semantic.get("intent"), "entity_ids": semantic.get("entity_ids", [])},
                ensure_ascii=False,
            )
        )

    return "{rules}\n\n{hint}CÂU HỎI: {question}\n\nCONTEXT:\n{context}\n\nCÂU TRẢ LỜI CÓ CĂN CỨ:".format(
        rules=SYSTEM_RULES, hint=hint, question=question, context=context
    )
