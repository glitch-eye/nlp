"""Build a deterministic, stratified human-review queue for generated labels.

The queue reduces review effort; it does not certify unreviewed rows as gold.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _tree_shape(parse_tree):
    """Keep nonterminal structure while ignoring terminals and leaf values."""
    if not parse_tree:
        return "NO_TREE"
    labels = re.findall(r"\(([A-Z][A-Z0-9_]*)", parse_tree)
    return "/".join(labels) if labels else "NO_TREE"


def _risk(row):
    reasons = []
    if row.get("parse_status") != "PARSED":
        reasons.append("parse_not_successful")
    if row.get("intent") in (None, "", "UNKNOWN"):
        reasons.append("unknown_or_missing_intent")
    if row.get("expected_answerability") == "EXPECTED_CONTEXT" and not row.get("expected_chunk_ids"):
        reasons.append("answerable_without_expected_chunks")
    if row.get("intent") not in (None, "UNKNOWN") and not row.get("entities"):
        reasons.append("intent_without_entities")
    if row.get("expected_answerability") == "REVIEW":
        reasons.append("answerability_requires_review")
    return reasons


def _group_key(row):
    entities = tuple(sorted(set(row.get("entities") or [])))
    chunk_count = len(row.get("expected_chunk_ids") or [])
    chunk_bucket = "0" if chunk_count == 0 else "1" if chunk_count == 1 else "2-5" if chunk_count <= 5 else "6+"
    return (
        row.get("intent", "UNKNOWN"),
        row.get("expected_answerability", "REVIEW"),
        _tree_shape(row.get("parse_tree")),
        entities,
        chunk_bucket,
    )


def _stable_rank(row):
    value = "{}\0{}".format(row.get("sample_id", ""), row.get("query", ""))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path):
    rows = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON at line {}: {}".format(line_no, error))
        if not isinstance(row, dict) or not isinstance(row.get("query"), str):
            raise ValueError("Line {} must contain an object with query".format(line_no))
        rows.append(row)
    return rows


def build_audit_queue(rows, per_stratum=2, max_samples=300):
    if per_stratum < 1:
        raise ValueError("per_stratum must be >= 1")
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")

    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    selected = []
    # Allocate at least one slot to each stratum when possible, then fill
    # remaining capacity proportionally by deterministic hash order.
    ordered_groups = sorted(groups.items(), key=lambda item: repr(item[0]))
    for key, members in ordered_groups:
        ranked = sorted(members, key=_stable_rank)
        for row in ranked[:per_stratum]:
            selected.append((key, row))

    if len(selected) > max_samples:
        selected.sort(key=lambda item: (0 if _risk(item[1]) else 1, _stable_rank(item[1])))
        selected = selected[:max_samples]
    elif len(selected) < max_samples:
        already = {row.get("sample_id") for _, row in selected}
        remaining = [
            (key, row) for key, members in ordered_groups for row in members
            if row.get("sample_id") not in already
        ]
        remaining.sort(key=lambda item: (0 if _risk(item[1]) else 1, _stable_rank(item[1])))
        selected.extend(remaining[:max_samples - len(selected)])

    queue = []
    for key, row in selected:
        queue.append({
            "sample_id": row.get("sample_id"),
            "query": row["query"],
            "auto_label": {
                "parse_status": row.get("parse_status"),
                "intent": row.get("intent"),
                "entities": row.get("entities", []),
                "expected_chunk_ids": row.get("expected_chunk_ids", []),
                "expected_answerability": row.get("expected_answerability"),
                "label_status": row.get("label_status", "UNSPECIFIED"),
            },
            "audit_group": hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:10],
            "risk_flags": _risk(row),
            "review": {
                "review_status": "PENDING",
                "correct_intent": None,
                "correct_entities": None,
                "correct_expected_chunk_ids": None,
                "correct_answerability": None,
                "reviewer_note": "",
            },
        })
    return queue


def main():
    app_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create a stratified label-review queue")
    parser.add_argument("--input", default=str(app_dir / "output" / "samples_labeled.jsonl"))
    parser.add_argument("--output", default=str(app_dir / "output" / "label_audit_queue.jsonl"))
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    queue = build_audit_queue(rows, args.per_stratum, args.max_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
        encoding="utf-8",
    )
    print("Input rows: {}".format(len(rows)))
    print("Audit rows: {}".format(len(queue)))
    print("Risk flags: {}".format(dict(Counter(flag for row in queue for flag in row["risk_flags"]))))
    print("Audit queue: {}".format(output))


if __name__ == "__main__":
    main()
