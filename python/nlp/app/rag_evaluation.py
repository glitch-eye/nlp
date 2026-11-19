"""Part IV evidence for the LLM/RAG branch (Member 2).

Everything in the report is MEASURED by running the assistant, so it is regenerated
(not hand-written) whenever the grammar, data, retriever or model changes:

* metrics on the labelled sample set (hit@k, refusal agreement, groundedness),
* live examples: grounded answers, questions the knowledge base cannot answer,
* hallucination probes: a SCRIPTED misbehaving model is plugged into the real pipeline to
  show what ``answer_guard`` does with fabricated dates, invented sources and uncited text.
  The scripted model is a test double, clearly labelled, never presented as a real LLM.

Usage (from python/):  python -m nlp.app.rag_evaluation [--output-dir DIR] [--provider mock|openai]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES = APP_DIR / "output" / "samples_labeled.jsonl"

GROUNDED_CASES = [
    "Môn NLP có bao nhiêu tín chỉ?",
    "Chương 4 nói về gì?",
    "Tuần 5 học những nội dung gì?",
]
NO_INFO_CASES = [
    ("câu ngoài miền (mẫu của văn phạm)", "Hôm nay ở TP.HCM có mưa không?"),
    ("thuộc tính không có trong dữ liệu", "Phòng học của môn CO3085 ở đâu?"),
    ("mốc thời gian không có trong dữ liệu", "Deadline của bài tập lớn là khi nào?"),
]
# Paraphrased out-of-domain questions the grammar does NOT list: how far does each layer get?
OOD_PROBES = [
    "Thời tiết hôm nay thế nào?", "Cho tôi công thức nấu phở", "Giá vàng hôm nay bao nhiêu?",
    "Messi đá cho đội nào?", "Bạn có thể viết giúp tôi một bài thơ không?",
    "Python 3.12 có gì mới?", "Làm sao để nấu cơm cho ngon?", "Ai là tổng thống Mỹ hiện nay?",
    "Tỷ giá đô la hôm nay thế nào?", "Phim nào đang chiếu rạp?",
]
REFUSALS = ("NO_RELEVANT_CONTEXT", "INSUFFICIENT_CONTEXT")

DEGRADED_WARNING = [
    "CẢNH BÁO: Knowledge Base đang ở trạng thái degraded hoặc không dùng KnowledgeBase chuẩn.",
    "Các kết quả retrieval/RAG trong báo cáo này có thể không đại diện cho dữ liệu môn học đầy đủ.",
]
def kb_is_degraded(assistant):
    """Return whether the assistant is using a non-standard/incomplete KB object."""
    from nlp.app.knowledge_base import KnowledgeBase
    return not isinstance(getattr(assistant, "kb", None), KnowledgeBase)


class ScriptedLLM:
    """Test double that returns a fixed kind of BAD answer. Not a real language model."""

    model_name = "SCRIPTED test model (not a real LLM)"

    def __init__(self, script):
        self.script = script
        self.last_output = None

    def generate(self, prompt, question="", chunks=None, semantic=None):
        self.last_output = self.script(chunks or [])
        return self.last_output


def _label(chunk):
    return "[Nguồn: {} / {}]".format(chunk["source"], chunk["section"])


HALLUCINATION_PROBES = [
    ("Mô hình bịa ngày hạn nộp nhưng trích dẫn đúng nguồn", "Bài tập lớn có những yêu cầu gì?",
     lambda ch: "Bài tập lớn gồm nhiều phần. Hạn nộp cuối cùng là 15/12/2026. " + _label(ch[0])),
    ("Mô hình trích dẫn nguồn không được gửi vào prompt", "Bài tập lớn có những yêu cầu gì?",
     lambda ch: "Bài tập lớn gồm nhiều phần. [Nguồn: assignments.txt / MỤC 9]"),
    ("Mô hình trả lời không trích dẫn nguồn", "Bài tập lớn có những yêu cầu gì?",
     lambda ch: "Bài tập lớn gồm nhiều phần."),
    ("Mô hình tuân thủ prompt, trả INSUFFICIENT_CONTEXT", "Python 3.12 có gì mới?",
     lambda ch: "INSUFFICIENT_CONTEXT"),
]


def _short(text, n=170):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _pct(a, b):
    return "{:.1f}%".format(100.0 * a / b) if b else "n/a"


# ---------------------------------------------------------------- labelled-set metrics
def load_labeled(path=DEFAULT_SAMPLES):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_labeled(assistant, rows):
    results = []
    for row in rows:
        out = assistant.answer(row["query"])
        results.append({
            "id": row.get("sample_id"), "query": row["query"], "intent": row.get("intent"),
            "expected": row["expected_answerability"],
            "expected_chunks": set(row.get("expected_chunk_ids") or []),
            "label_status": row.get("label_status"),
            "status": out["status"], "retrieved": out.get("retrieved_chunk_ids") or [],
            "sources": out.get("sources") or [], "answer": out.get("answer", ""),
            "citation": out.get("citation_check") or {},
        })
    return results


def labeled_metrics(results):
    with_ctx = [r for r in results if r["expected"] == "EXPECTED_CONTEXT"]
    hits = [r for r in with_ctx if r["expected_chunks"] & set(r["retrieved"])]
    answered = [r for r in results if r["status"] == "ANSWERED"]
    checked = [r for r in answered if r["citation"].get("status") == "ANSWERED"]
    return {
        "n": len(results),
        "confusion": Counter((r["expected"], r["status"]) for r in results),
        "labels": Counter(r["label_status"] for r in results),
        "with_ctx": len(with_ctx), "hits": len(hits),
        "retrieval_misses": [r for r in with_ctx if r not in hits],
        "answered": len(answered), "cited_ok": len(checked),
        "avg_sources": (sum(len(r["sources"]) for r in answered) / len(answered)) if answered else 0.0,
        "needs_review": [r for r in results if r["status"] == "NEEDS_REVIEW"],
        "answered_ok": sum(1 for r in with_ctx if r["status"] == "ANSWERED"),
        "n_insufficient": sum(1 for r in results if r["expected"] == "INSUFFICIENT_CONTEXT"),
        "refused_ok": sum(1 for r in results if r["expected"] != "EXPECTED_CONTEXT" and r["status"] in REFUSALS),
        "should_refuse": sum(1 for r in results if r["expected"] != "EXPECTED_CONTEXT"),
        "over_answered": [r for r in results if r["expected"] == "INSUFFICIENT_CONTEXT" and r["status"] == "ANSWERED"],
    }


# ---------------------------------------------------------------- live examples
def run_examples(assistant):
    grounded = [assistant.answer(q) for q in GROUNDED_CASES]
    no_info = [(kind, assistant.answer(q)) for kind, q in NO_INFO_CASES]
    ood = [(q, assistant.answer(q)) for q in OOD_PROBES]

    real_client = assistant.llm_client
    probes = []
    try:
        for title, question, script in HALLUCINATION_PROBES:
            sim = ScriptedLLM(script)
            assistant.llm_client = sim
            res = assistant.answer(question)
            probes.append((title, question, sim.last_output, res))
    finally:
        assistant.llm_client = real_client
    return {"grounded": grounded, "no_info": no_info, "ood": ood, "probes": probes}


# ---------------------------------------------------------------- rendering
def _cited(res):
    return ", ".join("{} / {}".format(s["source"], s["section"]) for s in res.get("sources") or []) or "(không có)"


def render_sections(metrics, examples, dialogue_results=None, model_name="", degraded=False):
    L = []
    add = L.append
    if degraded:
        L.extend(DEGRADED_WARNING)
        add("")
    if metrics:
        m = metrics
        add("2. Đánh giá diện rộng trên tập câu hỏi có nhãn (samples_labeled.jsonl) [ĐO TRỰC TIẾP]:")
        add("   - Tổng số mẫu kiểm thử                         : {}".format(m["n"]))
        add("   - Trạng thái nhãn                              : {}".format(
            ", ".join("{}={}".format(k, v) for k, v in m["labels"].items())))
        add("   - Hit@k (có chunk kỳ vọng trong top-k)         : {} / {} ({})".format(m["hits"], m["with_ctx"], _pct(m["hits"], m["with_ctx"])))
        add("   - Trả lời khi nhãn là EXPECTED_CONTEXT         : {} / {} ({})".format(m["answered_ok"], m["with_ctx"], _pct(m["answered_ok"], m["with_ctx"])))
        add("   - Từ chối khi nhãn yêu cầu từ chối             : {} / {} ({})".format(m["refused_ok"], m["should_refuse"], _pct(m["refused_ok"], m["should_refuse"])))
        add("   - Câu ANSWERED có trích dẫn hợp lệ             : {} / {} ({})".format(m["cited_ok"], m["answered"], _pct(m["cited_ok"], m["answered"])))
        add("   - Số nguồn trung bình mỗi câu ANSWERED         : {:.2f} (chỉ tính nguồn thực sự được trích dẫn)".format(m["avg_sources"]))
        add("   - Câu bị đánh dấu NEEDS_REVIEW                 : {}".format(len(m["needs_review"])))
        add("   - Ma trận (nhãn kỳ vọng -> trạng thái thực tế):")
        for (exp, got), cnt in sorted(m["confusion"].items()):
            add("       {:<22} -> {:<22} : {}".format(exp, got, cnt))
        add("   Lưu ý: nhãn do build_sample_dataset gán tự động (AUTO_LABEL_REQUIRES_REVIEW), chưa rà soát thủ công.")
    else:
        add("2. Đánh giá diện rộng: KHÔNG CÓ samples_labeled.jsonl nên bỏ qua (chạy build_sample_dataset trước).")
    add("")

    add("II. CÁC VÍ DỤ CHO NHÁNH LLM/RAG (sinh bằng cách chạy hệ thống; model = {})".format(model_name))
    add("-" * 70)
    add("1. Câu trả lời đúng và có căn cứ (Grounded Answer):")
    for res in examples["grounded"]:
        add("   - Câu hỏi   : \"{}\"".format(res["question"]))
        add("   - Truy xuất : {}".format(", ".join(res.get("retrieved_chunk_ids") or []) or "(không có)"))
        add("   - Trạng thái: {}   Nguồn được trích dẫn: {}".format(res["status"], _cited(res)))
        add("   - Trả lời   : \"{}\"".format(_short(res["answer"], 230)))
    add("")
    add("2. Câu hỏi không có thông tin trong Knowledge Base / ngoài phạm vi:")
    for kind, res in examples["no_info"]:
        add("   - [{}] \"{}\"".format(kind, res["question"]))
        add("       status={}, retrieval_status={}, sources={}".format(res["status"], res.get("retrieval_status"), _cited(res)))
        add("       Trả lời: \"{}\"".format(_short(res["answer"], 200)))
    add("")
    add("3. Nguy cơ sinh thông tin không có trong dữ liệu (Hallucination) - thử với model GIẢ LẬP có kịch bản:")
    add("   (Model giả lập chỉ dùng để kiểm tra answer_guard; KHÔNG phải kết quả của một LLM thật.)")
    for title, question, raw, res in examples["probes"]:
        cc = res.get("citation_check") or {}
        add("   - {} | Câu hỏi: \"{}\"".format(title, question))
        add("       Model trả về : \"{}\"".format(_short(raw, 200)))
        add("       Guard        : status={}, trích dẫn sai={}, thông tin không có trong nguồn={}".format(
            res["status"], cc.get("invalid_citations") or [], cc.get("unsupported_facts") or []))
        add("       Người dùng thấy: \"{}\"".format(_short(res["answer"], 200)))
    add("")

    add("III. TRƯỜNG HỢP SAI / HẠN CHẾ (liệt kê từ kết quả chạy thật)")
    add("-" * 70)
    n_ood = len(examples["ood"])
    refused = [q for q, r in examples["ood"] if r["status"] in REFUSALS]
    leaked = [(q, r) for q, r in examples["ood"] if r["status"] not in REFUSALS]
    add("1. Câu ngoài miền diễn đạt lại (không nằm trong mẫu của văn phạm): từ chối đúng {} / {}".format(len(refused), n_ood))
    for q, r in leaked:
        add("   - Vẫn bị trả lời [{}]: \"{}\" -> \"{}\"".format(r["status"], q, _short(r["answer"], 110)))
    if leaked:
        add("   Nguyên nhân: bộ lọc trước LLM chỉ dựa trên điểm truy xuất và độ phủ từ vựng; mock không đánh giá được")
        add("   sự liên quan. Với model thật, prompt yêu cầu trả INSUFFICIENT_CONTEXT và answer_guard xử lý (xem II.3, trường hợp cuối).")
    if metrics:
        add("2. Truy xuất sai (không có chunk kỳ vọng trong top-k): {} / {}".format(len(metrics["retrieval_misses"]), metrics["with_ctx"]))
        for r in metrics["retrieval_misses"][:5]:
            add("   - {} \"{}\": kỳ vọng {} ; nhận {}".format(r["id"], r["query"], sorted(r["expected_chunks"]), r["retrieved"]))
        over = metrics["over_answered"]
        add("3. Nhãn INSUFFICIENT_CONTEXT nhưng hệ thống vẫn trả lời: {} / {}".format(len(over), metrics["n_insufficient"]))
        by_intent = Counter(r["intent"] for r in over)
        add("   Phân bố theo intent: {}".format(dict(by_intent)))
        for r in over[:3]:
            add("   - \"{}\" -> trích dẫn: {}".format(r["query"], ", ".join("{} / {}".format(s["source"], s["section"]) for s in r["sources"]) or "(không có)"))
        add("   Cần rà soát thủ công: nếu nguồn được trích dẫn thực sự chứa đáp án thì lỗi nằm ở nhãn tự động chứ không ở hệ thống.")
    unparsed = [r for r in (dialogue_results or []) if r.get("parse_status") == "UNPARSED"]
    add("4. Câu hội thoại mà văn phạm Phần I không nhận (UNPARSED): {}".format(len(unparsed)))
    for r in unparsed:
        add("   - \"{}\" -> hệ thống vẫn truy xuất bằng câu gốc, status={}".format(r.get("resolved_query") or r.get("question"), r["status"]))
    return L


REAL_MODEL_SAMPLE_CAP = 50   # labelled questions sent to a paid API unless the caller asks for more


def is_mock(client):
    return "mock" in str(getattr(client, "model_name", "")).lower()


def _subsample(rows, limit):
    """Evenly strided subset (the labelled file is grouped by question form, so stride keeps coverage)."""
    if not limit or len(rows) <= limit:
        return rows
    step = len(rows) / float(limit)
    return [rows[int(i * step)] for i in range(limit)]


def build_rag_sections(assistant, dialogue_results=None, samples_path=DEFAULT_SAMPLES, max_samples=None):
    """Return the report lines. With a real (paid) model only REAL_MODEL_SAMPLE_CAP labelled
    questions are evaluated unless ``max_samples`` is given; the mock evaluates all of them."""
    rows = load_labeled(samples_path)
    if max_samples is None and not is_mock(assistant.llm_client):
        max_samples = REAL_MODEL_SAMPLE_CAP
    rows = _subsample(rows, max_samples)
    metrics = labeled_metrics(evaluate_labeled(assistant, rows)) if rows else None
    examples = run_examples(assistant)
    return render_sections(metrics, examples, dialogue_results,
                           getattr(assistant.llm_client, "model_name", ""), kb_is_degraded(assistant))


def main():
    from nlp.app.assistant import CourseAssistant, create_llm_client

    ap = argparse.ArgumentParser(description="Generate measured Part IV evidence for the RAG branch")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--output-dir", default=str(APP_DIR / "output"))
    ap.add_argument("--data-dir")
    ap.add_argument("--grammar")
    ap.add_argument("--provider", choices=("mock", "openai", "auto"))
    ap.add_argument("--model")
    ap.add_argument("--max-samples", type=int, help="Evaluate only N labelled questions (default: all for mock, 50 for a real model)")
    args = ap.parse_args()
    assistant = CourseAssistant(data_dir=args.data_dir, grammar_path=args.grammar,
                                llm_client=create_llm_client(provider=args.provider, model=args.model))
    lines = build_rag_sections(assistant, samples_path=args.samples, max_samples=args.max_samples)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (out / "rag_evaluation.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
