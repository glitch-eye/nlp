#!/usr/bin/env python3
"""Demo helper: shows, for each question, everything the assignment's demo video must show.

    1. the question the system receives
    2. parsing result (parse tree)                       -> requirement 3
    3. semantic frame (intent / entities)                -> requirement 3
    4. retrieval (top-k chunks with scores)              -> LLM/RAG requirement
    5. the context actually sent to the LLM              -> LLM/RAG requirement
    6. the answer, its citations, status and the model   -> requirement 4
    A question the system cannot answer shows status + "LLM not called"   -> requirement 5

Run from python/ (same environment variables as the real-model run):
    python demo_trace.py --tour                    # scripted scenes, press Enter between them
    python demo_trace.py "Tuần 7 học gì?"          # one question
    python demo_trace.py --interactive             # type questions; 'reset' clears context, 'exit' quits
    python demo_trace.py --tour --provider openai --model qwen2.5:7b
Add --full-prompt to print the whole prompt (rules included) instead of only the CONTEXT block.
"""
import argparse
import sys
import textwrap
from pathlib import Path

from nlp.app.assistant import CourseAssistant, create_llm_client
from nlp.app.dialogue import DialogueState

# Scenes chosen from questions that passed in the real-model evaluation run.
# Each scene is a list of questions asked in ONE dialogue (state is reset between scenes).
TOUR = [
    ("Câu hỏi thông tin môn học (grounded + citation)", ["Môn NLP có bao nhiêu tín chỉ?"]),
    ("Câu hỏi lịch học theo tuần", ["Tuần 7 học gì?"]),
    ("Câu hỏi đa ý định (multi-intent)", ["Tuần 3 học gì và chương 4 nói về gì?"]),
    ("Hội thoại nhiều lượt (giải tham chiếu 'chương sau')", ["Chương 4 nói về gì?", "Còn chương sau?"]),
    ("Hệ thống KHÔNG thể trả lời: thiếu dữ liệu trong knowledge base", ["Phòng học môn NLP ở đâu?"]),
    ("Hệ thống KHÔNG thể xử lý: tham chiếu không có tiền từ -> hỏi làm rõ", ["Nó là gì?"]),
]
BAR = "=" * 78


def hr(title):
    print("\n--- {} ".format(title).ljust(78, "-"))


def wrap(text, indent="    "):
    return textwrap.fill(" ".join(str(text).split()), width=100, initial_indent=indent, subsequent_indent=indent)


def parse_tree(assistant, text):
    try:
        tree = assistant.parser.parse(text)
    except Exception as exc:  # parser problems must not break the demo
        return None, "parser error: {}".format(type(exc).__name__)
    return (tree.to_bracketed() if tree is not None else None), None


def show_prompts(prompts, full):
    if not prompts:
        return False
    for i, prompt in enumerate(prompts, 1):
        if len(prompts) > 1:
            print("  [lời gọi LLM {}/{}]".format(i, len(prompts)))
        if full:
            print(textwrap.indent(prompt, "    | "))
            continue
        start = prompt.find("CONTEXT:")
        end = prompt.find("NHẮC LẠI")
        context = prompt[start:end if end > start else None].strip() if start >= 0 else prompt
        print(textwrap.indent(context, "    | "))
    return True


def trace(assistant, question, state, prompts, full_prompt=False):
    del prompts[:]  # the LLM wrapper refills this list during answer()
    print("\n" + BAR)
    print("QUÁ TRÌNH XỬ LÝ: Parser -> Semantic frame -> Retrieval -> Prompt + LLM -> Answer Guard")
    hr("1. CÂU HỎI ĐẦU VÀO")
    print("    {}".format(question))

    result = assistant.answer(question, dialogue_state=state)
    clauses = result.get("clauses") or []

    hr("2. [Parser] Earley parser + CFG -> parse tree")
    tree, err = parse_tree(assistant, question)
    resolved = result.get("resolved_query")
    resolved_tree = None
    if not tree and resolved and resolved != question:
        resolved_tree, _ = parse_tree(assistant, resolved)
    if tree:
        print(wrap(tree))
    elif clauses:
        print("    Câu đa ý định: tách thành {} mệnh đề, mỗi mệnh đề được parse riêng:".format(len(clauses)))
        for c in clauses:
            sub, _ = parse_tree(assistant, c.get("query", ""))
            print("    - {!r}".format(c.get("query")))
            print(wrap(sub or "UNPARSED", "        "))
    elif resolved_tree:
        print("    Câu gốc chứa tham chiếu; parse câu đã giải tham chiếu {!r}:".format(resolved))
        print(wrap(resolved_tree))
    elif result.get("parse_status") == "SKIPPED":
        print("    (bỏ qua parse: hệ thống cần hỏi làm rõ tham chiếu trước)")
    else:
        print("    UNPARSED {} - câu nằm ngoài văn phạm; hệ thống vẫn truy xuất bằng câu gốc.".format(err or ""))

    hr("3. [Semantic frame] intent và entity")
    print("    parse_status : {}".format(result.get("parse_status")))
    print("    intent       : {}".format(result.get("intent")))
    if result.get("intents"):
        print("    intents      : {}".format(result.get("intents")))
    print("    entities     : {}".format(result.get("entity_ids")))
    if resolved and resolved != question:
        print("    câu đã giải tham chiếu : {}".format(resolved))

    hr("4. [Retrieval] top-k chunk và điểm xếp hạng")
    print("    retrieval_status = {}".format(result.get("retrieval_status")))
    for chunk, score in zip(result.get("chunks") or [], result.get("scores") or []):
        print("    {:.3f}  {} / {}".format(score, chunk.get("source"), chunk.get("section")))
        print(wrap(chunk.get("text", "")[:160], "           "))
    if not result.get("chunks"):
        print("    (không có chunk nào đủ liên quan)")

    hr("5. [Prompt + LLM] context gửi cho mô hình")
    if not show_prompts(prompts, full_prompt):
        print("    LLM KHÔNG được gọi (bị chặn trước đó: status = {}).".format(result.get("status")))

    hr("6. [Answer Guard] kiểm tra và câu trả lời")
    check = result.get("citation_check")
    if check:
        print("    kiểm tra trích dẫn : {} hợp lệ, {} không hợp lệ".format(len(check.get("valid_citations") or []), len(check.get("invalid_citations") or [])))
    else:
        print("    Answer Guard dừng pipeline trước khi gọi LLM (không có gì để kiểm tra trích dẫn)")
    print("    status : {}".format(result.get("status")))
    print("    model  : {}".format(getattr(assistant.llm_client, "model_name", "unknown") if prompts else "(không gọi model)"))
    print(wrap(result.get("answer", "")))
    if result.get("sources"):
        print("    nguồn  : {}".format("; ".join("{} / {}".format(s.get("source"), s.get("section")) if isinstance(s, dict) else str(s) for s in result["sources"])))
    print(BAR)
    return result


def main():
    try:  # Windows consoles: make Vietnamese render correctly
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Demo trace for the Course Assistant")
    ap.add_argument("questions", nargs="*", help="questions asked in one dialogue")
    ap.add_argument("--tour", action="store_true", help="run the scripted demo scenes")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter between scenes")
    ap.add_argument("--full-prompt", action="store_true")
    ap.add_argument("--data-dir", default=None, help="course data folder (default: data/ next to this script)")
    ap.add_argument("--provider", default=None, help="mock | openai | auto (default: LLM_PROVIDER env, else mock)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = create_llm_client(provider=args.provider, model=args.model)
    prompts = []
    original = client.generate

    def recording_generate(prompt, *a, **kw):
        prompts.append(prompt)
        return original(prompt, *a, **kw)

    client.generate = recording_generate
    # Resolve relative to this file, not the current directory, so it runs from python/ or from the repo root.
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent / "data"
    if not data_dir.is_dir():
        sys.exit("Không tìm thấy thư mục dữ liệu: {}\nDùng --data-dir <đường dẫn tới thư mục data>.".format(data_dir))
    assistant = CourseAssistant(data_dir=str(data_dir), llm_client=client)
    print("LLM: {} | knowledge base: {} chunks".format(getattr(client, "model_name", "unknown"), len(assistant.kb.documents)))

    if args.tour:
        for number, (title, questions) in enumerate(TOUR, 1):
            print("\n########## Cảnh {}/{}: {}".format(number, len(TOUR), title))
            if not args.no_pause:
                input("[Enter để chạy cảnh này] ")
            state = DialogueState()
            for q in questions:
                trace(assistant, q, state, prompts, args.full_prompt)
        return
    if args.questions:
        state = DialogueState()
        for q in args.questions:
            trace(assistant, q, state, prompts, args.full_prompt)
        return
    # interactive (default)
    state = DialogueState()
    print("Nhập câu hỏi. 'reset' xóa ngữ cảnh hội thoại, 'exit' để thoát.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "reset":
            state = DialogueState()
            print("(đã xóa ngữ cảnh hội thoại)")
            continue
        if q:
            trace(assistant, q, state, prompts, args.full_prompt)


if __name__ == "__main__":
    main()
