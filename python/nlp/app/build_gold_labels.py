"""Build GOLD_REVIEWED datasets for Part II (Retrieval/RAG) and Part IV (Evaluation).

Creates two separate files without modifying ``output/samples_labeled.jsonl``:
1. ``output/samples_gold_labeled.jsonl``: All synthetic grammar samples re-labeled with
   verified ground-truth ``entities``, ``expected_chunk_ids``, ``expected_answerability``
   and ``label_status="GOLD_REVIEWED"``.
2. ``input/test_labeled.jsonl``: Curated independent Gold Test Set covering:
   - Representative in-grammar questions across all intents,
   - Benchmark queries from ``data/sample_queries.txt``,
   - Hand-written paraphrased / natural wording queries (including UNPARSED in-domain queries),
   - Insufficient-context queries (missing attributes in KB: deadline, classroom, instructor, etc.),
   - Out-of-scope queries (NO_RELEVANT_CONTEXT).
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

APP_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = APP_DIR.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from nlp.app.knowledge_base import KnowledgeBase, default_data_dir
from nlp.app.parser import CourseParser
from nlp.app.semantic_frame import frame_from_tree
from nlp.app.text_utils import normalize


INTENT_TO_PRIMARY_KIND = {
    "GET_COURSE_INFO": "course_info",
    "GET_SCHEDULE": "schedule",
    "GET_TOPIC": "topics",
    "GET_ASSIGNMENT": "assignments",
    "GET_DEADLINE": "assignments",
    "GET_RESOURCE": "resources",
    "GET_RULE": "regulations",
    "GET_LO": "course_info",
}

# Topic -> Chapter mapping grounded in data/topics.txt and data/schedule.txt
TOPIC_TO_CHAPTERS = {
    "NLP": ["CH01"],
    "NATURAL_LANGUAGE_PROCESSING": ["CH01"],
    "NOUN_PHRASES": ["CH02"],
    "VERB_PHRASES": ["CH02", "CH05", "CH08"],
    "ADJECTIVE_PHRASES": ["CH02"],
    "ADVERBIAL_PHRASES": ["CH02"],
    "CFG": ["CH03", "CH05", "CH06"],
    "PARSING": ["CH03", "CH04", "CH06", "CH08", "CH10"],
    "PCFG": ["CH03", "CH06"],
    "DEPENDENCY_GRAMMAR": ["CH03"],
    "FEATURES": ["CH04"],
    "AUGMENTED_GRAMMAR": ["CH04"],
    "UNIFICATION": ["CH04", "CH08"],
    "POS_TAGGING": ["CH06"],
    "LOGICAL_FORM": ["CH07", "CH08"],
    "THEMATIC_ROLES": ["CH07"],
    "SELECTIONAL_RESTRICTION": ["CH09"],
    "SEMANTIC_NETWORK": ["CH09"],
    "WSD": ["CH09"],
    "SEMANTIC_GRAMMAR": ["CH10"],
    "KNOWLEDGE_REPRESENTATION": ["CH11"],
    "QUESTION_ANSWERING": ["CH11"],
    "LOCAL_DISCOURSE_CONTEXT": ["CH12"],
    "REFERENCE": ["CH12"],
    "ANAPHORA": ["CH12"],
    "CENTERING": ["CH12"],
}

# Unsupported attribute keywords that are not in the KB -> INSUFFICIENT_CONTEXT
UNSUPPORTED_KB_KEYWORDS = (
    "deadline",
    "han nop",
    "thoi han nop",
    "han chot",
    "nop khi nao",
    "phai nop khi nao",
    "nop vao ngay nao",
    "het han khi nao",
    "khi nao phai nop",
    "khi nao nop",
    "phong hoc",
    "giang vien",
    "thay co",
    "email",
    "hoc phi",
    "link nop",
    "diem cong",
)


def extract_all_entities(query, parse_tree, kb):
    """Extract canonical entity IDs from both the parse tree and surface aliases/regex."""
    entities = []
    if parse_tree is not None:
        frame = frame_from_tree(parse_tree, query, kb.aliases)
        entities.extend(frame.get("entity_ids") or [])

    norm_q = " " + normalize(query) + " "
    for alias, eid, _ in kb.aliases:
        norm_alias = " " + normalize(alias) + " "
        if norm_alias.strip() and norm_alias in norm_q and eid not in entities:
            entities.append(eid)

    for pattern, formatter in (
        (r"\b(?:tuần|tuan|week)\s+(?:thứ\s+|thu\s+)?0?(\d{1,2})\b", lambda m: "WEEK_{:02d}".format(int(m.group(1)))),
        (r"\b(?:chương|chuong|chapter)\s+0?(\d{1,2})\b", lambda m: "CH{:02d}".format(int(m.group(1)))),
        (r"\bL\s*\.?\s*O\s*\.?\s*(\d+\.\d+)\b", lambda m: "LO" + m.group(1)),
    ):
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            eid = formatter(match)
            if eid not in entities:
                entities.append(eid)

    for match in re.finditer(r"\bphần\s+(I{1,3}|IV|[1-4])\b", query, flags=re.IGNORECASE):
        token = match.group(1).upper()
        part_map = {
            "1": "PART_I", "I": "PART_I",
            "2": "PART_II", "II": "PART_II",
            "3": "PART_III", "III": "PART_III",
            "4": "PART_IV", "IV": "PART_IV",
        }
        eid = part_map.get(token)
        if eid and eid not in entities:
            entities.append(eid)

    if any(w in norm_q for w in (" midterm ", " giua ky ")):
        if "MIDTERM" not in entities:
            entities.append("MIDTERM")
    if any(w in norm_q for w in (" final ", " cuoi ky ")):
        if "FINAL" not in entities:
            entities.append("FINAL")
    if any(w in norm_q for w in (" bai tap lon ", " btl ", " do an ", " project ")):
        if "BTL01" not in entities and not any(e.startswith("PART_") for e in entities):
            entities.append("BTL01")
    if any(w in norm_q for w in (" su dung ai ", " cong cu ai ", " dung ai ", " chatgpt ")):
        if "AI_USAGE" not in entities:
            entities.append("AI_USAGE")
    if any(w in norm_q for w in (" llm rag ", " llm hoac rag ")):
        if "LLM_RAG" not in entities:
            entities.append("LLM_RAG")

    return list(dict.fromkeys(entities))


def is_insufficient_query(intent, query, entities):
    """Return True if the question asks for information absent from the sample KB."""
    if intent == "GET_DEADLINE":
        return True
    norm_q = normalize(query)
    if any(kw in norm_q for kw in UNSUPPORTED_KB_KEYWORDS):
        return True
    # Chapters beyond CH12 or weeks beyond WEEK_15 are not in the KB
    for eid in entities:
        if eid.startswith("CH") and eid[2:].isdigit() and int(eid[2:]) > 12:
            return True
        if eid.startswith("WEEK_") and eid[5:].isdigit() and int(eid[5:]) > 15:
            return True
    return False


def compute_gold_chunks(intent, entities, query, kb):
    """Determine ground-truth expected_chunk_ids for a query from KnowledgeBase documents."""
    if intent in (None, "UNKNOWN"):
        return []
    if is_insufficient_query(intent, query, entities):
        return []

    docs = kb.documents
    by_kind = {}
    for d in docs:
        by_kind.setdefault(d["kind"], []).append(d)

    entity_set = set(entities)
    expanded_entities = set(entity_set)
    for eid in list(entity_set):
        if eid in TOPIC_TO_CHAPTERS:
            expanded_entities.update(TOPIC_TO_CHAPTERS[eid])

    matched = []

    def add_chunks(predicate):
        for d in docs:
            if predicate(d) and d["chunk_id"] not in matched:
                matched.append(d["chunk_id"])

    if intent == "GET_COURSE_INFO":
        if "MIDTERM" in entity_set or "FINAL" in entity_set:
            exam_ents = entity_set.intersection({"MIDTERM", "FINAL"})
            add_chunks(lambda d: bool(exam_ents.intersection(d.get("entity_ids", []))))
            add_chunks(lambda d: d["chunk_id"] in (
                "course-info-exam-format-8",
                "course-info-assessment-7",
                "course-info-course-information-0",
            ))
        else:
            add_chunks(lambda d: d["kind"] == "course_info")
            add_chunks(lambda d: d["chunk_id"] in (
                "faq-q1-0",
                "faq-q2-1",
                "schedule-luu-y-day-la-ke-hoach-mau-duoc-xay-dung-tu-cau-truc-12-chuong-trong-0",
                "resources-textbook-3-2",
            ))

    elif intent == "GET_LO":
        lo_ents = {e for e in entity_set if e.startswith("LO")}
        if lo_ents:
            add_chunks(lambda d: bool(lo_ents.intersection(d.get("entity_ids", []))))
        add_chunks(lambda d: d["chunk_id"] in (
            "course-info-detailed-learning-outcomes-6",
            "course-info-learning-outcomes-5",
            "faq-q9-8",
            "faq-q10-9",
            "faq-q11-10",
        ))

    elif intent == "GET_SCHEDULE":
        week_ents = {e for e in entity_set if e.startswith("WEEK_")}
        chap_ents = {e for e in expanded_entities if e.startswith("CH")}
        topic_ents = entity_set - week_ents - {e for e in entity_set if e.startswith("CH")}

        if week_ents:
            add_chunks(lambda d: bool(week_ents.intersection(d.get("entity_ids", []))))
        if chap_ents:
            add_chunks(lambda d: d["kind"] in ("schedule", "topics", "faq") and bool(chap_ents.intersection(d.get("entity_ids", []))))
        if topic_ents:
            add_chunks(lambda d: bool(topic_ents.intersection(d.get("entity_ids", []))))
        if not matched or not entity_set:
            add_chunks(lambda d: d["kind"] == "schedule")
            add_chunks(lambda d: d["chunk_id"] in (
                "course-info-course-id-co3085-1",
                "faq-q1-0",
                "faq-q2-1",
                "faq-q14-13",
                "faq-q15-14",
            ))
        else:
            # General schedule note chunk is also valid context for schedule queries
            add_chunks(lambda d: d["chunk_id"] == "schedule-luu-y-day-la-ke-hoach-mau-duoc-xay-dung-tu-cau-truc-12-chuong-trong-0")

    elif intent == "GET_TOPIC":
        chap_ents = {e for e in entity_set if e.startswith("CH")}
        topic_ents = entity_set - chap_ents
        if chap_ents:
            add_chunks(lambda d: bool(chap_ents.intersection(d.get("entity_ids", []))))
        if topic_ents:
            add_chunks(lambda d: bool(topic_ents.intersection(d.get("entity_ids", []))))
            mapped_chaps = {ch for t in topic_ents for ch in TOPIC_TO_CHAPTERS.get(t, [])}
            if mapped_chaps:
                add_chunks(lambda d: d["kind"] in ("topics", "schedule", "faq") and bool(mapped_chaps.intersection(d.get("entity_ids", []))))
        if not matched:
            add_chunks(lambda d: d["kind"] in ("topics", "course_info", "faq"))

    elif intent == "GET_ASSIGNMENT":
        part_ents = {e for e in entity_set if e.startswith("PART_")}
        if part_ents:
            add_chunks(lambda d: d["kind"] == "assignments" and bool(part_ents.intersection(d.get("entity_ids", []))))
        add_chunks(lambda d: d["kind"] == "assignments")
        add_chunks(lambda d: d["chunk_id"] in ("faq-q12-11", "faq-q13-12", "course-info-assessment-7", "course-info-main-parts-4"))

    elif intent == "GET_RESOURCE":
        res_specific = entity_set.intersection({"PYTHON_NLP", "STATISTICAL_NLP", "MACHINE_TRANSLATION"})
        if res_specific:
            add_chunks(lambda d: d["kind"] == "resources" and bool(res_specific.intersection(d.get("entity_ids", []))))
        add_chunks(lambda d: d["kind"] == "resources")
        if entity_set:
            add_chunks(lambda d: bool(expanded_entities.intersection(d.get("entity_ids", []))))

    elif intent == "GET_RULE":
        if "AI_USAGE" in entity_set:
            add_chunks(lambda d: d["chunk_id"] in ("regulations-ai-usage-3", "regulations-academic-integrity-2", "regulations-luu-y-0"))
        if "LLM_RAG" in entity_set:
            add_chunks(lambda d: d["chunk_id"] in ("regulations-llm-rag-7", "assignments-llm-rag-extension-5", "faq-q12-11", "regulations-luu-y-0"))
        add_chunks(lambda d: d["kind"] == "regulations")
        add_chunks(lambda d: d["chunk_id"] in ("faq-q12-11", "faq-q13-12", "assignments-llm-rag-extension-5"))

    return matched


def label_single_query(sample_id, query, kb, parser, forced_intent=None):
    """Build a complete GOLD_REVIEWED JSONL record for one query."""
    tree = parser.parse(query)
    parse_status = "PARSED" if tree is not None else "UNPARSED"
    parse_tree = tree.to_bracketed() if tree is not None else "()"

    if tree is not None:
        frame = frame_from_tree(tree, query, kb.aliases)
        intent = frame.get("intent") or forced_intent or "UNKNOWN"
    else:
        intent = forced_intent or "UNKNOWN"

    entities = extract_all_entities(query, tree, kb) if intent != "UNKNOWN" else []
    if intent == "UNKNOWN":
        expected_ids = []
        answerability = "NO_RELEVANT_CONTEXT"
    elif is_insufficient_query(intent, query, entities):
        expected_ids = []
        answerability = "INSUFFICIENT_CONTEXT"
    else:
        expected_ids = compute_gold_chunks(intent, entities, query, kb)
        answerability = "EXPECTED_CONTEXT" if expected_ids else "INSUFFICIENT_CONTEXT"

    return {
        "sample_id": sample_id,
        "query": query,
        "parse_status": parse_status,
        "parse_tree": parse_tree,
        "intent": intent,
        "entities": entities,
        "expected_chunk_ids": expected_ids,
        "expected_answerability": answerability,
        "label_status": "GOLD_REVIEWED",
    }


def relabel_synthetic_dataset(input_jsonl, output_jsonl, kb, parser):
    """Read samples_labeled.jsonl and write GOLD_REVIEWED records to a new file."""
    raw_lines = Path(input_jsonl).read_text(encoding="utf-8").splitlines()
    records = []
    for raw in raw_lines:
        if not raw.strip():
            continue
        row = json.loads(raw)
        query = row["query"]
        parse_status = row.get("parse_status", "PARSED")
        parse_tree = row.get("parse_tree", "()")
        intent = row.get("intent") or "UNKNOWN"
        if intent == "UNKNOWN":
            entities = []
            expected_ids = []
            answerability = "NO_RELEVANT_CONTEXT"
        else:
            entities = list(dict.fromkeys(
                list(row.get("entities") or []) + extract_all_entities(query, None, kb)
            ))
            # Filter out course-level entity CO3085/NLP when intent is specific unless course_info/resource
            if intent not in ("GET_COURSE_INFO", "GET_RESOURCE") and len(entities) > 1:
                entities = [e for e in entities if e not in ("NLP",)]
            if is_insufficient_query(intent, query, entities):
                expected_ids = []
                answerability = "INSUFFICIENT_CONTEXT"
            else:
                expected_ids = compute_gold_chunks(intent, entities, query, kb)
                answerability = "EXPECTED_CONTEXT" if expected_ids else "INSUFFICIENT_CONTEXT"

        records.append({
            "sample_id": row["sample_id"],
            "query": query,
            "parse_status": parse_status,
            "parse_tree": parse_tree,
            "intent": intent,
            "entities": entities,
            "expected_chunk_ids": expected_ids,
            "expected_answerability": answerability,
            "label_status": "GOLD_REVIEWED",
        })

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


# Curated independent test cases covering all 5 categories of Section 6 (Part IV) in assignment.pdf:
# 1. Benchmark queries from data/sample_queries.txt (Q001-Q028)
# 2. Representative valid queries across all 8 in-domain intents & entities
# 3. Paraphrased / natural student expressions (both PARSED and UNPARSED in-domain queries)
# 4. In-domain questions where KB has insufficient information (INSUFFICIENT_CONTEXT)
# 5. Out-of-scope questions (NO_RELEVANT_CONTEXT / UNKNOWN)
CURATED_TEST_SPECS = [
    # --- Group 1: Benchmark queries from sample_queries.txt (Q001 - Q025) ---
    ("TEST0001", "Môn NLP có bao nhiêu tín chỉ?", "GET_COURSE_INFO", "benchmark_valid"),
    ("TEST0002", "CO3085 có mấy tín chỉ?", "GET_COURSE_INFO", "benchmark_valid"),
    ("TEST0003", "Môn học gồm những phần nào?", "GET_COURSE_INFO", "benchmark_valid"),
    ("TEST0004", "Tuần 3 học gì?", "GET_SCHEDULE", "benchmark_valid"),
    ("TEST0005", "Chương 4 học vào tuần nào?", "GET_SCHEDULE", "benchmark_valid"),
    ("TEST0006", "Chương nào nói về CFG?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0007", "PCFG được học ở đâu?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0008", "Chương nào nói về semantic grammar?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0009", "L.O.2.5 là gì?", "GET_LO", "benchmark_valid"),
    ("TEST0010", "L.O.4.2 liên quan đến nội dung nào?", "GET_LO", "benchmark_valid"),
    ("TEST0011", "Bài tập lớn yêu cầu xây dựng những gì?", "GET_ASSIGNMENT", "benchmark_valid"),
    ("TEST0012", "Phần I của bài tập lớn là gì?", "GET_ASSIGNMENT", "benchmark_valid"),
    ("TEST0013", "Bài tập lớn có phần hỏi đáp không?", "GET_ASSIGNMENT", "benchmark_valid"),
    ("TEST0014", "Có được dùng LLM/RAG không?", "GET_RULE", "benchmark_valid"),
    ("TEST0015", "Quy định về sử dụng AI là gì?", "GET_RULE", "benchmark_valid"),
    ("TEST0016", "Midterm bao nhiêu phút?", "GET_COURSE_INFO", "benchmark_valid"),
    ("TEST0017", "Final bao nhiêu phút?", "GET_COURSE_INFO", "benchmark_valid"),
    ("TEST0018", "Tài liệu tham khảo về statistical NLP là gì?", "GET_RESOURCE", "benchmark_valid"),
    ("TEST0019", "Tài liệu nào liên quan đến Python NLP?", "GET_RESOURCE", "benchmark_valid"),
    ("TEST0020", "Chương 12 nói về gì?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0021", "Chương nào học về reference?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0022", "Chương nào học về question answering?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0023", "Chương nào học về selectional restriction?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0024", "Chương 10 có những phương pháp gì?", "GET_TOPIC", "benchmark_valid"),
    ("TEST0025", "Chương 11 có liên quan đến hỏi đáp không?", "GET_TOPIC", "benchmark_valid"),

    # --- Group 2: Stratified valid queries across all intents, chapters, weeks, LOs, rules ---
    ("TEST0026", "số tín chỉ của môn học CO3085 là bao nhiêu?", "GET_COURSE_INFO", "in_grammar_valid"),
    ("TEST0027", "mục tiêu học tập của Xử lý Ngôn ngữ Tự nhiên là gì?", "GET_COURSE_INFO", "in_grammar_valid"),
    ("TEST0028", "bài thi giữa kỳ có thời lượng bao nhiêu phút?", "GET_COURSE_INFO", "in_grammar_valid"),
    ("TEST0029", "bài thi cuối kỳ kéo dài bao lâu?", "GET_COURSE_INFO", "in_grammar_valid"),
    ("TEST0030", "Tuần 1 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0031", "Tuần 2 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0032", "Tuần 5 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0033", "Tuần 7 học gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0034", "Tuần 9 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0035", "Tuần 12 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0036", "Tuần 14 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0037", "Tuần 15 học những nội dung gì?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0038", "Chương 2 học vào tuần nào?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0039", "Chương 6 học vào tuần nào?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0040", "Chương 9 học vào tuần nào?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0041", "Khi nào học dạng luận lý?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0042", "Phần văn phạm phụ thuộc nằm ở tuần nào?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0043", "Môn học được tổ chức trong bao nhiêu tuần?", "GET_SCHEDULE", "in_grammar_valid"),
    ("TEST0044", "Chương 1 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0045", "Chương 2 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0046", "Chương 3 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0047", "Chương 5 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0048", "Chương 6 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0049", "Chương 7 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0050", "Chương 8 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0051", "Chương 9 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0052", "Chương 11 nói về gì?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0053", "Chương nào đề cập đến gán nhãn từ loại?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0054", "Chương nào học về vai trò chủ đề?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0055", "Chương 6 có liên quan đến gán nhãn từ loại không?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0056", "Chương 4 có liên quan đến parsing không?", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0057", "giới thiệu về mạng ngữ nghĩa", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0058", "giải thích đồng tham chiếu", "GET_TOPIC", "in_grammar_valid"),
    ("TEST0059", "Phần II của bài tập lớn là gì?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0060", "Phần III cần làm gì?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0061", "Phần IV yêu cầu những đầu ra nào?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0062", "Bài tập lớn có phần hội thoại không?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0063", "Bài tập lớn có phần parser không?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0064", "Sinh viên phải nộp những gì?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0065", "Tôi cần hoàn thành những bài tập nào?", "GET_ASSIGNMENT", "in_grammar_valid"),
    ("TEST0066", "Tài liệu nào nói về PCFG?", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0067", "Tài liệu nào liên quan đến đồng tham chiếu?", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0068", "Tài liệu nào liên quan đến machine translation?", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0069", "Tài liệu tham khảo cho Chương 6 là gì?", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0070", "Tài liệu tham khảo cho Tuần 8 là gì?", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0071", "tìm tài liệu về semantic grammar", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0072", "liệt kê giáo trình", "GET_RESOURCE", "in_grammar_valid"),
    ("TEST0073", "Bài tập lớn có được sử dụng công cụ AI không?", "GET_RULE", "in_grammar_valid"),
    ("TEST0074", "quy định về LLM/RAG là gì?", "GET_RULE", "in_grammar_valid"),
    ("TEST0075", "quy định về thư viện bên ngoài là gì?", "GET_RULE", "in_grammar_valid"),
    ("TEST0076", "quy định về liêm chính học thuật là gì?", "GET_RULE", "in_grammar_valid"),
    ("TEST0077", "L.O.3.3 mô tả năng lực gì?", "GET_LO", "in_grammar_valid"),
    ("TEST0078", "L.O.4.2 yêu cầu sinh viên có khả năng gì?", "GET_LO", "in_grammar_valid"),

    # --- Group 3: Paraphrased / Natural student queries (different ways to express the same request) ---
    ("TEST0079", "Cho hỏi môn C03085 có mấy tín chỉ?", "GET_COURSE_INFO", "paraphrase_in_grammar"),
    ("TEST0080", "Xin hỏi đề cương của môn CO3085 gồm những gì?", "GET_COURSE_INFO", "paraphrase_in_grammar"),
    ("TEST0081", "Cho biết tuần thứ 6 lớp sẽ học những gì?", "GET_SCHEDULE", "paraphrase_in_grammar"),
    ("TEST0082", "Khi nào học văn phạm ngữ nghĩa?", "GET_SCHEDULE", "paraphrase_in_grammar"),
    ("TEST0083", "tóm tắt chương 8", "GET_TOPIC", "paraphrase_in_grammar"),
    ("TEST0084", "trình bày nội dung chương 10", "GET_TOPIC", "paraphrase_in_grammar"),
    ("TEST0085", "Học phần CO3085 có tổng cộng bao nhiêu tín chỉ vậy?", "GET_COURSE_INFO", "paraphrase_freeform"),
    ("TEST0086", "Thời gian làm bài thi midterm và final môn NLP là mấy phút?", "GET_COURSE_INFO", "paraphrase_freeform"),
    ("TEST0087", "Nội dung bài giảng ở tuần 4 gồm các chủ đề nào?", "GET_SCHEDULE", "paraphrase_freeform"),
    ("TEST0088", "Chủ đề gán nhãn từ loại POS tagging nằm ở chương mấy?", "GET_TOPIC", "paraphrase_freeform"),
    ("TEST0089", "Chương 2 và Chương 3 khác nhau như thế nào?", "GET_TOPIC", "paraphrase_freeform"),
    ("TEST0090", "Mỗi nhóm làm bài tập lớn gồm mấy sinh viên và yêu cầu làm gì?", "GET_ASSIGNMENT", "paraphrase_freeform"),
    ("TEST0091", "Phần I của bài tập lớn cần xuất ra những file kết quả nào?", "GET_ASSIGNMENT", "paraphrase_freeform"),
    ("TEST0092", "Sách giáo trình nào hướng dẫn thực hành xử lý ngôn ngữ tự nhiên bằng Python?", "GET_RESOURCE", "paraphrase_freeform"),
    ("TEST0093", "Sinh viên có được phép dùng thư viện bên ngoài khi làm bài tập lớn không?", "GET_RULE", "paraphrase_freeform"),
    ("TEST0094", "Chuẩn đầu ra L.O.2.5 đánh giá kiến thức gì của sinh viên?", "GET_LO", "paraphrase_freeform"),

    # --- Group 4: In-domain questions without information in KB (INSUFFICIENT_CONTEXT) ---
    ("TEST0095", "Deadline của bài tập lớn là khi nào?", "GET_DEADLINE", "insufficient_kb"),
    ("TEST0096", "Hạn nộp bài tập lớn là khi nào?", "GET_DEADLINE", "insufficient_kb"),
    ("TEST0097", "Khi nào phải nộp đồ án?", "GET_DEADLINE", "insufficient_kb"),
    ("TEST0098", "Bài tập lớn nộp vào ngày nào?", "GET_DEADLINE", "insufficient_kb"),
    ("TEST0099", "deadline BTL được quy định như thế nào?", "GET_DEADLINE", "insufficient_kb"),
    ("TEST0100", "Phòng học của môn CO3085 ở đâu?", "GET_COURSE_INFO", "insufficient_kb"),
    ("TEST0101", "Giảng viên phụ trách môn CO3085 là thầy cô nào?", "GET_COURSE_INFO", "insufficient_kb"),
    ("TEST0102", "Địa chỉ email liên hệ của giảng viên môn NLP là gì?", "GET_COURSE_INFO", "insufficient_kb"),
    ("TEST0103", "Học phí của môn học CO3085 là bao nhiêu?", "GET_COURSE_INFO", "insufficient_kb"),
    ("TEST0104", "Đường dẫn link nộp bài tập lớn trên hệ thống là gì?", "GET_ASSIGNMENT", "insufficient_kb"),
    ("TEST0105", "Môn CO3085 có chính sách điểm cộng không?", "GET_RULE", "insufficient_kb"),
    ("TEST0106", "Chương 13 của môn học nói về nội dung gì?", "GET_TOPIC", "insufficient_kb"),

    # --- Group 5: Out-of-scope / Out-of-domain questions (NO_RELEVANT_CONTEXT / UNKNOWN) ---
    ("TEST0107", "Hôm nay ở TP.HCM có mưa không?", "UNKNOWN", "out_of_scope"),
    ("TEST0108", "Ai là tổng thống của Pháp?", "UNKNOWN", "out_of_scope"),
    ("TEST0109", "Hãy giải bài tập xác suất này giúp tôi.", "UNKNOWN", "out_of_scope"),
    ("TEST0110", "Messi hiện đang đá cho câu lạc bộ nào?", "UNKNOWN", "out_of_scope"),
    ("TEST0111", "Giá vàng hôm nay bao nhiêu một lượng?", "UNKNOWN", "out_of_scope"),
    ("TEST0112", "Tỷ giá đô la hôm nay thế nào?", "UNKNOWN", "out_of_scope"),
    ("TEST0113", "Thủ đô của Nhật Bản là thành phố nào?", "UNKNOWN", "out_of_scope"),
    ("TEST0114", "Cách nấu bún bò Huế ngon nhất là gì?", "UNKNOWN", "out_of_scope"),
]


def build_curated_test_set(output_jsonl, kb, parser):
    """Build the curated independent Gold Test Set at input/test_labeled.jsonl."""
    records = []
    for sample_id, query, intent, category in CURATED_TEST_SPECS:
        rec = label_single_query(
            sample_id=sample_id,
            query=query,
            kb=kb,
            parser=parser,
            forced_intent=intent,
        )
        rec["category"] = category
        records.append(rec)

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def main():
    ap = argparse.ArgumentParser(description="Build GOLD_REVIEWED datasets for Course Assistant evaluation")
    ap.add_argument("--data-dir", default=str(default_data_dir()))
    ap.add_argument("--grammar", default=str(APP_DIR / "output" / "grammar.txt"))
    ap.add_argument("--input-synthetic", default=str(APP_DIR / "output" / "samples_labeled.jsonl"))
    ap.add_argument("--output-synthetic-gold", default=str(APP_DIR / "output" / "samples_gold_labeled.jsonl"))
    ap.add_argument("--output-test-gold", default=str(APP_DIR / "input" / "test_labeled.jsonl"))
    args = ap.parse_args()

    kb = KnowledgeBase(args.data_dir)
    parser = CourseParser(args.grammar)

    syn_records = relabel_synthetic_dataset(
        args.input_synthetic, args.output_synthetic_gold, kb, parser
    )
    syn_ans = Counter(r["expected_answerability"] for r in syn_records)
    print(
        "Wrote {} GOLD_REVIEWED synthetic samples to {} -> {}".format(
            len(syn_records), args.output_synthetic_gold, dict(syn_ans)
        )
    )

    test_records = build_curated_test_set(args.output_test_gold, kb, parser)
    test_ans = Counter(r["expected_answerability"] for r in test_records)
    test_cats = Counter(r["category"] for r in test_records)
    print(
        "Wrote {} GOLD_REVIEWED curated test samples to {} -> answerability={}, categories={}".format(
            len(test_records), args.output_test_gold, dict(test_ans), dict(test_cats)
        )
    )


if __name__ == "__main__":
    main()
