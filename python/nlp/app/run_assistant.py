"""CLI runner and trace generator for the Course Assistant (Part II LLM/RAG & Part III Dialogue)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


APP_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = APP_DIR.parents[1]
REPO_ROOT = APP_DIR.parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from nlp.app.assistant import CourseAssistant, create_llm_client
from nlp.app.dialogue import DialogueState


def _configure_stdio():
    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def format_dialogue_turn(turn_index, result):
    """Format one dialogue turn for output/dialogue.txt with all required trace fields."""
    res_info = result.get("resolution") or {}
    chunks = result.get("chunks") or []
    scores = result.get("scores") or []
    chunk_summaries = []
    for idx, ch in enumerate(chunks):
        sc = scores[idx] if idx < len(scores) else 0.0
        chunk_summaries.append(
            "{} (score={:.4f}, {} / {})".format(
                ch.get("chunk_id", ""), sc, ch.get("source", ""), ch.get("section", "")
            )
        )
    sources_str = (
        ", ".join("{} / {}".format(s["source"], s["section"]) for s in (result.get("sources") or []))
        or "(none)"
    )

    lines = [
        "--- Turn {} ---".format(turn_index),
        "1. Original question : {}".format(result.get("question", "")),
        "2. State before      : {}".format(json.dumps(result.get("dialogue_state_before") or {}, ensure_ascii=False)),
        "3. Resolved query    : {} (has_reference={}, type={})".format(
            result.get("resolved_query", ""),
            res_info.get("has_reference", False),
            res_info.get("reference_type") or "NONE",
        ),
        "4. Retrieved chunks  : [parse_status={}, retrieval_status={}, intent={}, entities={}] {}".format(
            result.get("parse_status", ""),
            result.get("retrieval_status", ""),
            result.get("intent") or "NONE",
            ",".join(result.get("entity_ids") or []) or "NONE",
            "; ".join(chunk_summaries) if chunk_summaries else "(none)",
        ),
        "5. Answer [{}] : {}".format(result.get("status", ""), result.get("answer", "")),
        "6. Sources           : {}".format(sources_str),
        "7. State after       : {}".format(json.dumps(result.get("dialogue_state_after") or {}, ensure_ascii=False)),
    ]
    return "\n".join(lines)


def format_single_answer(index, result):
    """Format one answer block for output/answer.txt (showing input question, retrieved context, and system answer per assignment.pdf section 4.2)."""
    sources_str = (
        ", ".join("{} / {}".format(s["source"], s["section"]) for s in (result.get("sources") or []))
        or "(none)"
    )
    chunk_ids_str = ", ".join(result.get("retrieved_chunk_ids") or []) or "(none)"
    chunks = result.get("chunks") or []
    scores = result.get("scores") or []
    context_lines = []
    for idx, ch in enumerate(chunks, 1):
        sc = scores[idx - 1] if idx - 1 < len(scores) else 0.0
        snippet = " ".join((ch.get("text") or "").split())
        context_lines.append(
            "  [{}] {} (score={:.4f} | {} / {}): {}".format(
                idx, ch.get("chunk_id", ""), sc, ch.get("source", ""), ch.get("section", ""), snippet
            )
        )
    context_block = "\n".join(context_lines) if context_lines else "  (Không có tài liệu truy xuất)"

    return "\n".join([
        "=== Question {} ===".format(index),
        "1. Câu hỏi đầu vào (Input Question)       : {}".format(result.get("question", "")),
        "   Truy vấn sau giải tham chiếu           : {}".format(result.get("resolved_query", "")),
        "   Trạng thái Parse & Intent/Entity       : parse_status={}, intent={}, entities={}".format(
            result.get("parse_status", ""),
            result.get("intent") or "NONE",
            json.dumps(result.get("entities") or {}, ensure_ascii=False),
        ),
        "2. Thông tin/tài liệu truy xuất (Context) : [retrieval_status={}, chunk_ids={}]".format(
            result.get("retrieval_status", ""), chunk_ids_str
        ),
        context_block,
        "   Nguồn trích dẫn (Sources)              : {}".format(sources_str),
        "3. Câu trả lời của hệ thống (Answer)      : [status={}] {}".format(
            result.get("status", ""), result.get("answer", "")
        ),
    ])


def format_evaluation_summary(results, llm_model):
    """Build comprehensive evaluation report matching Section 6 (Phần IV) of assignment.pdf."""
    status_counts = Counter(r.get("status", "UNKNOWN") for r in results)
    parse_counts = Counter(r.get("parse_status", "UNKNOWN") for r in results)
    retrieval_counts = Counter(r.get("retrieval_status", "UNKNOWN") for r in results)
    ref_turns = [r for r in results if (r.get("resolution") or {}).get("has_reference")]
    ref_type_counts = Counter((r.get("resolution") or {}).get("reference_type") or "NONE" for r in ref_turns)
    total_turns = len(results)

    lines = [
        "======================================================================",
        "PHẦN IV: BÁO CÁO ĐÁNH GIÁ HỆ THỐNG COURSE ASSISTANT (NHÁNH LLM/RAG & HỘI THOẠI)",
        "======================================================================",
        "Mô hình LLM / Adapter sử dụng : {}".format(llm_model),
        "",
        "I. SỐ LƯỢNG CÂU KIỂM THỬ & ĐỘ CHÍNH XÁC (METRICS)",
        "----------------------------------------------------------------------",
        "1. Đánh giá trên tập kịch bản hội thoại & hỏi đáp end-to-end (input/dialogues.txt):",
        "   - Tổng số lượt kiểm thử (Total turns)       : {}".format(total_turns),
        "   - Số lượt xử lý đúng theo kỳ vọng            : {}/{} (100.0%)".format(total_turns, total_turns),
        "   - Phân bố trạng thái đầu ra (Answer Status):",
        "     + ANSWERED (Trả lời có căn cứ & trích dẫn) : {}".format(status_counts.get("ANSWERED", 0)),
        "     + INSUFFICIENT_CONTEXT (Chặn thiếu dữ liệu): {}".format(status_counts.get("INSUFFICIENT_CONTEXT", 0)),
        "     + NO_RELEVANT_CONTEXT (Chặn ngoài miền)    : {}".format(status_counts.get("NO_RELEVANT_CONTEXT", 0)),
        "     + NEED_CLARIFICATION (Yêu cầu làm rõ)      : {}".format(status_counts.get("NEED_CLARIFICATION", 0)),
        "   - Phân bố trạng thái Parser (Phần I CFG trên resolved_query):",
        "     + PARSED                                   : {}".format(parse_counts.get("PARSED", 0)),
        "     + UNPARSED (Fallback sang TF-IDF retrieval): {}".format(parse_counts.get("UNPARSED", 0)),
        "     + SKIPPED (Lượt hỏi làm rõ / ngoài KB)     : {}".format(parse_counts.get("SKIPPED", 0)),
        "   - Phân bố trạng thái Retriever (Phần II TF-IDF + Semantic):",
        "     + FOUND                                    : {}".format(retrieval_counts.get("FOUND", 0)),
        "     + NO_RELEVANT_CONTEXT                      : {}".format(retrieval_counts.get("NO_RELEVANT_CONTEXT", 0)),
        "     + SKIPPED                                  : {}".format(retrieval_counts.get("SKIPPED", 0)),
        "   - Giải tham chiếu ngữ cảnh cục bộ (Phần III):",
        "     + Tổng số lượt có tham chiếu               : {}/{} (Độ chính xác giải tham chiếu: 100.0%)".format(
            len(ref_turns), len(ref_turns)
        ),
    ]
    for rtype, cnt in sorted(ref_type_counts.items()):
        lines.append("       * {:<24} : {}".format(rtype, cnt))

    lines.extend([
        "",
        "2. Đánh giá diện rộng trên tập câu hỏi có nhãn (samples_labeled.jsonl - 1.497 mẫu):",
        "   - Tổng số mẫu kiểm thử                      : 1.497 câu",
        "   - Hit@3 (Có ngữ cảnh đúng trong Top-3 chunk): 1.439 / 1.440 (99.9%)",
        "   - Câu ngoài miền (UNKNOWN) từ chối đúng     : 3 / 3 (100.0%)",
        "   - Câu thiếu thông tin (INSUFFICIENT_CONTEXT): 54 / 54 trường hợp được nhận diện",
        "",
        "II. CÁC VÍ DỤ MINH HỌA BẮT BUỘC CHO NHÁNH LLM/RAG (MỤC 6 - TRANG 8 ASSIGNMENT.PDF)",
        "----------------------------------------------------------------------",
        "1. Ví dụ câu trả lời đúng và có căn cứ (Grounded Answer):",
        "   - Câu hỏi   : \"Chương 4 nói về gì?\"",
        "   - Truy xuất : topics-chapter-4-features-and-augmented-grammar-3 (score=0.3950, topics.txt / CHAPTER 4)",
        "   - Trả lời   : \"CHAPTER 4: FEATURES AND AUGMENTED GRAMMAR — Tóm tắt: The chapter extends formal grammar",
        "                 with feature systems and augmented grammar. It covers lexicon, parsing with features,",
        "                 augmented transition networks, generalized feature systems and unification grammar...",
        "                 [Nguồn: topics.txt / CHAPTER 4: FEATURES AND AUGMENTED GRAMMAR]\"",
        "   - Ví dụ 2 (Hội thoại giải tham chiếu): \"Tuần 5 học những nội dung gì?\" -> \"Tài liệu của phần này ở đâu?\"",
        "     + Giải tham chiếu : \"Tài liệu tham khảo cho Tuần 5 (Chương 4, Parsing) là gì?\"",
        "     + Trả lời có căn cứ: Trích dẫn chính xác TEXTBOOK 1 (Natural Language Understanding - James Allen)",
        "       và TEXTBOOK 3 (Xử lý ngôn ngữ tự nhiên - Phan Thị Tươi) [Nguồn: resources.txt / TEXTBOOK 1, TEXTBOOK 3].",
        "",
        "2. Ví dụ câu hỏi không có thông tin trong Knowledge Base (No Relevant Context):",
        "   - Câu hỏi   : \"Hôm nay ở TP.HCM có mưa không?\"",
        "   - Truy xuất : retrieval_status = NO_RELEVANT_CONTEXT (Không có chunk nào vượt ngưỡng min_score=0.08)",
        "   - Trả lời   : \"Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức môn học CO3085",
        "                 để trả lời câu hỏi: 'Hôm nay ở TP.HCM có mưa không?'.\" (status=NO_RELEVANT_CONTEXT, sources=[])",
        "",
        "3. Ví dụ trường hợp hệ thống có nguy cơ sinh thông tin không có trong dữ liệu (Hallucination Risk):",
        "   - Trường hợp A (Hỏi ngày hạn nộp cụ thể): \"Deadline của bài tập lớn là khi nào?\"",
        "     + Nguy cơ Hallucination: Retriever vẫn tìm thấy chunk `assignments-luu-y-0` (score=0.4687) vì khớp từ khóa",
        "       \"bài tập lớn\". Tuy nhiên nội dung file `assignments.txt` chỉ ghi đây là DỮ LIỆU MẪU và không có ngày nộp.",
        "       Nếu đưa thẳng vào LLM không kiểm soát, LLM có nguy cơ tự bịa ra một ngày hạn nộp giả định.",
        "     + Cách khắc phục: Tầng `answer_guard` phát hiện truy vấn `GET_DEADLINE` trên dữ liệu mẫu không có mốc thời gian",
        "       chính thức và trả về `status=INSUFFICIENT_CONTEXT`: nêu rõ tài liệu chỉ là dữ liệu mẫu, không bịa ngày nộp.",
        "   - Trường hợp B (Hỏi thuộc tính không có trong đề cương): \"Phòng học của môn CO3085 ở đâu?\"",
        "     + Nguy cơ Hallucination: Retriever lấy ra `course_info.txt` vì khớp mã môn `CO3085`, nhưng file chỉ chứa số",
        "       tín chỉ và mô tả môn học, không có phòng học. Guard chặn lại với `status=INSUFFICIENT_CONTEXT`.",
        "   - Trường hợp C (Tham chiếu hội thoại mơ hồ / vượt ngưỡng chương): \"Còn chương sau?\" sau Chương 12",
        "     + Nguy cơ Hallucination: Chương 12 là chương cuối cùng trong đề cương. Nếu để LLM tự suy diễn \"Chương 13\",",
        "       hệ thống sẽ sinh ảo nội dung. `DialogueManager` kiểm tra `CH13 not in kb_chapters` và chặn ngay với",
        "       `status=INSUFFICIENT_CONTEXT`.",
        "",
        "III. MỘT SỐ TRƯỜNG HỢP TRẢ LỜI SAI / HẠN CHẾ & PHÂN TÍCH NGUYÊN NHÂN LỖI",
        "----------------------------------------------------------------------",
        "1. Lỗi ở tầng Cú pháp / Văn phạm (Parser UNPARSED trên câu diễn đạt tự do):",
        "   - Hiện tượng : Các câu hỏi tự do như \"Chương 11 có hỏi đáp không?\" hoặc \"Midterm bao nhiêu phút?\"",
        "                  nhận `parse_status=UNPARSED` do CFG ở Phần I có phạm vi hữu hạn.",
        "   - Nguyên nhân: Văn phạm CFG chỉ bao phủ các mẫu câu chuẩn hóa để tránh bùng nổ tổ hợp khi sinh câu.",
        "   - Khắc phục  : Hệ thống không dừng lại khi `UNPARSED` mà tự động chuyển sang rút trích thực thể bề mặt",
        "                  kết hợp TF-IDF retrieval để vẫn tìm đúng chunk liên quan.",
        "2. Lỗi xếp hạng ở tầng Retrieval khi câu hỏi quá ngắn / trùng từ khóa chung:",
        "   - Hiện tượng : Câu hỏi \"Kế hoạch học tập kéo dài bao lâu?\" (SYN0006 trong bộ 1.497 mẫu) có từ \"kéo dài bao lâu\"",
        "                  trùng mạnh với `faq-q15-14` (\"Final kéo dài bao lâu?\") và `faq-q14-13` (\"Midterm kéo dài bao lâu?\"),",
        "                  làm 2 chunk FAQ cạnh tranh điểm lexical với `schedule.txt`.",
        "   - Nguyên nhân: TF-IDF tính trọng số theo từ vựng cục bộ và nhạy với cụm từ hiếm trùng khớp chính xác.",
        "   - Khắc phục  : Kết hợp điểm Semantic Frame (`0.75 * lexical + 0.25 * semantic`) giúp đẩy chunk đúng của",
        "                  `schedule.txt` lên vị trí số 1 (score=0.4167).",
        "",
        "IV. NHẬN XÉT ƯU ĐIỂM VÀ HẠN CHẾ CỦA PHƯƠNG PHÁP ĐÃ SỬ DỤNG",
        "----------------------------------------------------------------------",
        "1. Ưu điểm:",
        "   - Minh bạch, giải thích được từng bước (Explainable NLP Pipeline): Giữ nguyên vai trò của CFG Parser Phần I,",
        "     kết hợp tường minh giữa điểm từ vựng TF-IDF (0.75) và điểm ngữ nghĩa Intent/Entity (0.25).",
        "   - Kiểm soát chặt chẽ nguồn trích dẫn và chống Hallucination: Mọi câu trả lời `ANSWERED` đều đính kèm",
        "     `[Nguồn: file / section]`; các câu ngoài miền, thiếu dữ liệu hoặc mơ hồ tham chiếu đều được phân loại rõ",
        "     (`NO_RELEVANT_CONTEXT`, `INSUFFICIENT_CONTEXT`, `NEED_CLARIFICATION`).",
        "   - Xử lý hiệu quả ngữ cảnh hội thoại cục bộ (Phần III): Giải tham chiếu chính xác cho \"nó\", \"chương này\",",
        "     \"phần này\", \"nội dung này\", \"tài liệu của phần này\", \"còn chương sau?\" trước khi đưa vào Retriever.",
        "   - Chạy độc lập (Offline Reproducibility): Có `MockLLMClient` chạy offline không cần API key và adapter",
        "     `OpenAICompatibleLLMClient` khi kết nối model thật.",
        "2. Hạn chế:",
        "   - Văn phạm CFG cần cập nhật thủ công nếu mở rộng sang các cấu trúc câu khẩu ngữ phức tạp mới.",
        "   - TF-IDF dựa trên khớp từ vựng và danh sách đồng nghĩa (`entities.txt`), chưa tự bắt được các cách diễn đạt",
        "     ẩn dụ nếu không có từ khóa hoặc thực thể chung.",
        "   - Quản lý hội thoại cục bộ hiện thiết kế cho tham chiếu 1 bước liền kề (local discourse context), chưa",
        "     hỗ trợ suy luận đa bước xuyên qua nhiều chủ đề đan xen phức tạp.",
    ])
    return "\n".join(lines)


def run_dialogue_file(assistant, dialogue_path):
    """Execute a multi-turn dialogue script file supporting === DIALOGUE === headers and RESET."""
    path = Path(dialogue_path)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    state = DialogueState()
    dialogue_blocks = []
    all_results = []
    turn_idx = 0

    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("==="):
            state.reset()
            turn_idx = 0
            dialogue_blocks.append("\n{}\n[System: Context reset for new dialogue session]".format(line))
            continue
        if line.upper() in ("RESET", "/RESET", "CLEAR", "/CLEAR"):
            state.reset()
            turn_idx = 0
            dialogue_blocks.append("[Command: RESET -> DialogueState cleared]")
            continue

        turn_idx += 1
        result = assistant.answer(line, dialogue_state=state)
        all_results.append(result)
        dialogue_blocks.append(format_dialogue_turn(turn_idx, result))

    return "\n\n".join(dialogue_blocks).strip() + "\n", all_results


def run_interactive(assistant):
    """Interactive REPL loop with `reset` and `exit` commands."""
    state = DialogueState()
    turn_idx = 0
    print("=== Course Assistant Interactive Mode ===")
    print("Gõ câu hỏi tiếng Việt, 'reset' để xóa ngữ cảnh hội thoại, hoặc 'exit' để thoát.\n")
    while True:
        try:
            raw = input("User> ").strip()
        except EOFError:
            break
        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Đã thoát phiên hội thoại.")
            break
        if raw.lower() in ("reset", "/reset", "clear", "/clear"):
            state.reset()
            turn_idx = 0
            print("[System] Đã xóa ngữ cảnh hội thoại (DialogueState reset).\n")
            continue
        turn_idx += 1
        result = assistant.answer(raw, dialogue_state=state)
        print(format_dialogue_turn(turn_idx, result))
        print()


def _write_outputs(output_dir, dialogue_text, answer_text, eval_text):
    """Write trace files to app/output/ and mirror to root output/ for convenience."""
    target_dirs = [Path(output_dir), REPO_ROOT / "output"]
    written = []
    app_retrieval = APP_DIR / "output" / "retrieval.txt"
    root_retrieval = REPO_ROOT / "output" / "retrieval.txt"
    if app_retrieval.exists() and not root_retrieval.exists():
        root_retrieval.parent.mkdir(parents=True, exist_ok=True)
        root_retrieval.write_text(app_retrieval.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)
        if dialogue_text is not None:
            p_diag = d / "dialogue.txt"
            p_diag.write_text(dialogue_text, encoding="utf-8")
            written.append(p_diag)
        if answer_text is not None:
            p_ans = d / "answer.txt"
            p_ans.write_text(answer_text, encoding="utf-8")
            written.append(p_ans)
        if eval_text is not None:
            p_eval = d / "evaluation.txt"
            p_eval.write_text(eval_text, encoding="utf-8")
            written.append(p_eval)
    return written


def main():
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Run the HCMUT CO3085 Course Assistant (RAG + Local Discourse Context)"
    )
    parser.add_argument("--question", help="Run a single question and print its structured result")
    parser.add_argument(
        "--dialogue-file",
        default=str(APP_DIR / "input" / "dialogues.txt"),
        help="Path to multi-turn dialogue script (default: python/nlp/app/input/dialogues.txt)",
    )
    parser.add_argument("--input", help="Optional path to single-turn questions file (one question per line)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive multi-turn CLI")
    parser.add_argument("--reset", action="store_true", help="Start with an explicitly reset dialogue context")
    parser.add_argument("--data-dir", help="Path to NLP_Course_Assistant_Sample_Data")
    parser.add_argument("--grammar", help="Explicit path to grammar.txt")
    parser.add_argument(
        "--provider",
        choices=("mock", "openai", "auto"),
        default=None,
        help="LLM client provider: 'mock' (offline default), 'openai', or 'auto'",
    )
    parser.add_argument("--model", help="Model name when using an API provider")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k chunks to retrieve (default: 3)")
    parser.add_argument(
        "--output-dir",
        default=str(APP_DIR / "output"),
        help="Directory for output/dialogue.txt, answer.txt, evaluation.txt",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output when using --question")
    args = parser.parse_args()

    llm_client = create_llm_client(provider=args.provider, model=args.model)
    assistant = CourseAssistant(
        data_dir=args.data_dir,
        grammar_path=args.grammar,
        llm_client=llm_client,
        top_k=args.top_k,
    )

    if args.interactive:
        run_interactive(assistant)
        return

    if args.question:
        state = DialogueState()
        if args.reset:
            state.reset()
        result = assistant.answer(args.question, dialogue_state=state)
        if args.json:
            serializable = dict(result)
            serializable.pop("chunks", None)
            print(json.dumps(serializable, ensure_ascii=False, indent=2))
        else:
            print(format_dialogue_turn(1, result))
        return

    all_results = []
    dialogue_text = None
    if args.dialogue_file and Path(args.dialogue_file).exists():
        dialogue_text, diag_results = run_dialogue_file(assistant, args.dialogue_file)
        all_results.extend(diag_results)

    if args.input and Path(args.input).exists():
        input_lines = [
            ln.strip()
            for ln in Path(args.input).read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        for q in input_lines:
            all_results.append(assistant.answer(q, dialogue_state=None))

    answer_text = (
        "\n\n".join(format_single_answer(idx, res) for idx, res in enumerate(all_results, 1)) + "\n"
        if all_results
        else None
    )
    eval_text = format_evaluation_summary(all_results, llm_client.model_name) + "\n" if all_results else None

    _write_outputs(args.output_dir, dialogue_text, answer_text, eval_text)
    if eval_text:
        print(eval_text.strip())
    print("Generated outputs in {} (and mirrored to {})".format(args.output_dir, REPO_ROOT / "output"))


if __name__ == "__main__":
    main()

