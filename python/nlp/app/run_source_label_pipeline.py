"""Rebuild and label course-assistant data against the local knowledge base."""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from nlp.app.build_sample_dataset import build_records
from nlp.app.knowledge_base import KnowledgeBase, default_data_dir
from nlp.app.label_audit import apply_reviewed_labels, build_audit_queue, load_jsonl


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def chunk_ids_for_kind(documents, kind):
    return [doc["chunk_id"] for doc in documents if doc["kind"] == kind]


def find_chunk_ids(documents, kind, section_terms=None, entity_ids=None):
    candidates = [doc for doc in documents if doc["kind"] == kind]

    if entity_ids:
        entity_set = set(entity_ids)
        matched = [
            doc for doc in candidates
            if entity_set.intersection(set(doc.get("entity_ids", [])))
        ]
        if matched:
            candidates = matched

    if section_terms:
        terms = [normalize(term) for term in section_terms]
        matched = [
            doc for doc in candidates
            if any(term in normalize(doc["section"]) for term in terms)
        ]
        if matched:
            candidates = matched

    return [doc["chunk_id"] for doc in candidates]


def label_queue_row(row, documents):
    query = row["query"]
    normalized_query = normalize(query)
    intent = row["intent"]
    entities = list(row.get("entities") or [])

    if intent == "UNKNOWN":
        return [], "NO_RELEVANT_CONTEXT"
    if intent in ("GET_DEADLINE", "GET_INSTRUCTOR"):
        return [], "INSUFFICIENT_CONTEXT"

    if intent == "GET_COURSE_INFO":
        if "tin chi" in normalized_query:
            chunks = find_chunk_ids(documents, "course_info", ["course id"])
        elif "mo ta" in normalized_query:
            chunks = find_chunk_ids(documents, "course_info", ["course_description"])
        elif "muc tieu" in normalized_query:
            chunks = find_chunk_ids(documents, "course_info", ["Mục tiêu"])
        elif any(term in normalized_query for term in ("noi dung", "trang bi", "hoc duoc")):
            chunks = find_chunk_ids(
                documents,
                "course_info",
                ["main parts", "course_description", "learning_outcomes"],
            )
        elif any(term in normalized_query for term in ("danh gia", "thi ", "midterm", "final")):
            chunks = find_chunk_ids(documents, "course_info", ["assessment", "exam_format"])
        elif re.search(r"L\.?O\.?\s*\d+\.\d+", query, re.I):
            chunks = find_chunk_ids(documents, "course_info", ["detailed_learning_outcomes"])
        else:
            chunks = chunk_ids_for_kind(documents, "course_info")
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    if intent == "GET_SCHEDULE":
        week_match = re.search(r"(?<!\d)(?:tuần|week)\s*0*(\d{1,2})", query, re.I)
        if week_match:
            week_number = int(week_match.group(1))
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "schedule"
                and "week {:02d}".format(week_number) in normalize(doc["section"])
            ]
        elif entities:
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "schedule"
                and set(entities).intersection(set(doc.get("entity_ids", [])))
            ]
        else:
            chunks = chunk_ids_for_kind(documents, "schedule")
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    if intent == "GET_TOPIC":
        chapter_match = re.search(r"(?<!\d)(?:chương|chapter)\s*0*(\d{1,2})", query, re.I)
        if chapter_match:
            chapter_number = int(chapter_match.group(1))
            chapter_marker = "chapter {}".format(chapter_number)
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "topics"
                and chapter_marker in normalize(doc["text"])
            ]
        elif re.search(r"L\.?O\.?\s*\d+\.\d+", query, re.I):
            chunks = []
        elif entities:
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "topics"
                and set(entities).intersection(set(doc.get("entity_ids", [])))
            ]
        else:
            chunks = chunk_ids_for_kind(documents, "topics")
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    if intent == "GET_ASSIGNMENT":
        part_match = re.search(
            r"(?:phần|part)\s*(i|ii|iii|iv)\b",
            normalized_query,
            re.I,
        )
        if part_match:
            roman = part_match.group(1).upper()
            chunks = find_chunk_ids(documents, "assignments", ["PART {}".format(roman)])
        elif entities:
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "assignments"
                and set(entities).intersection(set(doc.get("entity_ids", [])))
            ]
        else:
            chunks = chunk_ids_for_kind(documents, "assignments")
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    if intent == "GET_RESOURCE":
        if entities:
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "resources"
                and set(entities).intersection(set(doc.get("entity_ids", [])))
            ]
        else:
            chunks = chunk_ids_for_kind(documents, "resources")
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    if intent == "GET_RULE":
        terms = []
        if "diem danh" in normalized_query or "attendance" in normalized_query:
            terms.append("attendance")
        if "trung thuc" in normalized_query or "academic" in normalized_query:
            terms.append("academic_integrity")
        if "ai" in normalized_query or "llm" in normalized_query:
            terms.extend(("AI_usage", "LLM_RAG"))
        if "ma nguon" in normalized_query or "source code" in normalized_query:
            terms.append("source_code")
        if "chia se" in normalized_query:
            terms.append("assignment_sharing")
        if "thu vien" in normalized_query:
            terms.append("external_libraries")
        if "ngoai mien" in normalized_query or "scope" in normalized_query:
            terms.append("OUT_OF_SCOPE")
        chunks = find_chunk_ids(documents, "regulations", terms)
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    if intent == "GET_LO":
        learning_outcome_match = re.search(r"L\.?O\.?\s*(\d+\.\d+)", query, re.I)
        if learning_outcome_match:
            learning_outcome = learning_outcome_match.group(1)
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "course_info"
                and learning_outcome in normalize(doc["text"])
            ]
        else:
            chunks = [
                doc["chunk_id"] for doc in documents
                if doc["kind"] == "course_info"
                and (
                    "learning outcomes" in normalize(doc["section"])
                    or "detailed_learning_outcomes" in normalize(doc["section"])
                )
            ]
        return chunks, "EXPECTED_CONTEXT" if chunks else "INSUFFICIENT_CONTEXT"

    return list(row.get("expected_chunk_ids") or []), row["expected_answerability"]


def source_label_queue(queue, source_rows, documents):
    source_by_id = {row["sample_id"]: row for row in source_rows}
    for item in queue:
        row = source_by_id[item["sample_id"]]
        chunks, answerability = label_queue_row(row, documents)
        review = item["review"]
        review["review_status"] = "REVIEWED"
        review["correct_parse_status"] = row.get("parse_status")
        review["correct_parse_tree"] = (
            row.get("parse_tree") if row.get("parse_status") == "PARSED" else None
        )
        review["correct_intent"] = row["intent"]
        review["correct_entities"] = list(row.get("entities") or [])
        review["correct_expected_chunk_ids"] = chunks
        review["correct_answerability"] = answerability
        review["reviewer_note"] = (
            "Đối chiếu với dữ liệu nguồn trong python/data/ và metadata chunk."
        )
    return queue


def main():
    app_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Tái tạo sample, gán nhãn theo dữ liệu nguồn và xuất tập reviewed",
    )
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--grammar", default=str(app_dir / "output" / "grammar.txt"))
    parser.add_argument("--samples", default=str(app_dir / "output" / "samples_labeled.jsonl"))
    parser.add_argument("--samples-text", default=str(app_dir / "output" / "samples.txt"))
    parser.add_argument("--queue", default=str(app_dir / "output" / "label_audit_queue.jsonl"))
    parser.add_argument("--reviewed", default=str(app_dir / "output" / "samples_reviewed.jsonl"))
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--max-sentences", type=int, default=10000)
    parser.add_argument(
        "--rebuild-samples",
        action="store_true",
        help="Sinh lại samples_labeled.jsonl từ grammar trước khi gán nhãn",
    )
    args = parser.parse_args()

    if args.rebuild_samples:
        records = build_records(
            args.grammar,
            args.data_dir,
            max_sentences=args.max_sentences,
        )
        write_jsonl(args.samples, records)
        samples_text = Path(args.samples_text)
        samples_text.parent.mkdir(parents=True, exist_ok=True)
        samples_text.write_text(
            "".join(record["query"] + "\n" for record in records),
            encoding="utf-8",
        )
        print("Rebuilt samples: {}".format(len(records)))

    source_rows = load_jsonl(args.samples)
    queue = build_audit_queue(source_rows, args.per_stratum, args.max_samples)

    knowledge_base = KnowledgeBase(args.data_dir)
    queue = source_label_queue(queue, source_rows, knowledge_base.documents)
    write_jsonl(args.queue, queue)

    reviewed_rows = apply_reviewed_labels(source_rows, queue)
    for row in reviewed_rows:
        row["label_provenance"] = "data_source_review"
        row["reviewer_note"] = (
            "Đối chiếu với dữ liệu nguồn trong python/data/ và metadata chunk."
        )
    write_jsonl(args.reviewed, reviewed_rows)

    print("Input rows: {}".format(len(source_rows)))
    print("Audit rows: {}".format(len(queue)))
    print("Reviewed gold rows: {}".format(len(reviewed_rows)))
    print("Answerability: {}".format(dict(Counter(
        item["review"]["correct_answerability"] for item in queue
    ))))
    print("Audit queue: {}".format(args.queue))
    print("Reviewed dataset: {}".format(args.reviewed))


if __name__ == "__main__":
    main()
