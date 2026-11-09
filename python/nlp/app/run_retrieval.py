"""Run retrieval against the hand-labeled query set or generated JSONL samples."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from nlp.app.knowledge_base import KnowledgeBase, default_data_dir
from nlp.app.retriever import Retriever, format_trace


def read_queries(path):
    """Read the pipe-delimited hand-labeled query file."""
    queries = []
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("=") or line.startswith("-") or "|" not in line:
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 2 and fields[0].startswith("Q"):
            queries.append({
                "sample_id": fields[0],
                "query": fields[1],
                "intent": fields[2] if len(fields) > 2 else None,
                "expected_entities": [fields[3]] if len(fields) > 3 else [],
            })
    return queries


def read_labeled_jsonl(path):
    """Load generated samples and validate the fields needed for retrieval evaluation."""
    samples = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            sample = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON on line {}: {}".format(line_number, error))
        if not isinstance(sample, dict) or not isinstance(sample.get("query"), str):
            raise ValueError("Line {} must contain a string 'query'".format(line_number))
        sample.setdefault("sample_id", "JSONL{:04d}".format(line_number))
        sample.setdefault("intent", "UNKNOWN")
        sample.setdefault("entities", [])
        sample.setdefault("expected_chunk_ids", [])
        sample.setdefault("expected_answerability", "REVIEW")
        samples.append(sample)
    return samples


def _retrieved_records(result):
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "score": round(score, 6),
            "source": chunk["source"],
            "section": chunk["section"],
        }
        for chunk, score in zip(result["chunks"], result["scores"])
    ]


def evaluate_samples(samples, retriever, top_k, use_gold_entities=False, bypass_unknown=False):
    cases = []
    expected_cases = []
    unknown_cases = []
    insufficient_cases = []
    by_intent = defaultdict(lambda: {
        "sample_count": 0,
        "expected_context_count": 0,
        "hit_at_k_count": 0,
        "unknown_bypassed_count": 0,
        "insufficient_context_count": 0,
        "insufficient_context_retrieved_count": 0,
    })

    for sample in samples:
        intent = sample.get("intent", "UNKNOWN")
        metrics = by_intent[intent]
        metrics["sample_count"] += 1
        semantic = ({"entity_ids": sample["entities"], "intent": intent}
                    if use_gold_entities else None)
        if intent == "UNKNOWN" and bypass_unknown:
            # The semantic layer routes out-of-domain questions around retrieval.
            result = {
                "query": sample["query"],
                "chunks": [],
                "scores": [],
                "retrieval_status": "NO_RELEVANT_CONTEXT",
            }
            metrics["unknown_bypassed_count"] += 1
        else:
            result = retriever.retrieve(sample["query"], semantic=semantic, top_k=top_k)
        retrieved = _retrieved_records(result)
        retrieved_ids = {record["chunk_id"] for record in retrieved}
        expected_ids = list(sample.get("expected_chunk_ids") or [])
        answerability = sample.get("expected_answerability", "REVIEW")
        hit_at_k = None

        if answerability == "EXPECTED_CONTEXT":
            hit_at_k = bool(retrieved_ids.intersection(expected_ids))
            expected_cases.append(hit_at_k)
            metrics["expected_context_count"] += 1
            metrics["hit_at_k_count"] += int(hit_at_k)
        elif sample.get("intent") == "UNKNOWN" or answerability == "NO_RELEVANT_CONTEXT":
            unknown_cases.append(intent == "UNKNOWN" and not retrieved)
        elif answerability == "INSUFFICIENT_CONTEXT":
            found = result["retrieval_status"] == "FOUND"
            insufficient_cases.append(found)
            metrics["insufficient_context_count"] += 1
            metrics["insufficient_context_retrieved_count"] += int(found)

        cases.append({
            "sample_id": sample["sample_id"],
            "query": sample["query"],
            "intent": sample.get("intent"),
            "entities": sample.get("entities", []),
            "expected_answerability": answerability,
            "expected_chunk_ids": expected_ids,
            "retrieval_status": result["retrieval_status"],
            "hit_at_k": hit_at_k,
            "retrieved": retrieved,
        })

    summary = {
        "sample_count": len(samples),
        "top_k": top_k,
        "min_score": retriever.min_score,
        "query_mode": "gold_semantic_frame_as_hint" if use_gold_entities else "original_query_only",
        "expected_context_count": len(expected_cases),
        "hit_at_k_count": sum(expected_cases),
        "hit_at_k": (sum(expected_cases) / float(len(expected_cases))) if expected_cases else None,
        "no_relevant_context_count": len(unknown_cases),
        "unknown_rejected_count": sum(unknown_cases),
        "unknown_rejection_rate": (sum(unknown_cases) / float(len(unknown_cases))) if unknown_cases else None,
        "insufficient_context_count": len(insufficient_cases),
        "insufficient_context_retrieved_count": sum(insufficient_cases),
        "by_intent": dict(by_intent),
    }
    return {"summary": summary, "cases": cases}


def format_evaluation_trace(evaluation):
    summary = evaluation["summary"]
    lines = [
        "Labeled sample retrieval evaluation",
        "Sample count: {}".format(summary["sample_count"]),
        "top_k: {}, min_score: {}".format(summary["top_k"], summary["min_score"]),
        "query_mode: {}".format(summary["query_mode"]),
        "Hit@{}: {}/{} ({})".format(
            summary["top_k"],
            summary["hit_at_k_count"],
            summary["expected_context_count"],
            "n/a" if summary["hit_at_k"] is None else "{:.1%}".format(summary["hit_at_k"]),
        ),
        "UNKNOWN rejected: {}/{} ({})".format(
            summary["unknown_rejected_count"],
            summary["no_relevant_context_count"],
            "n/a" if summary["unknown_rejection_rate"] is None else "{:.1%}".format(summary["unknown_rejection_rate"]),
        ),
        "INSUFFICIENT_CONTEXT cases with retrieval: {}/{}".format(
            summary["insufficient_context_retrieved_count"],
            summary["insufficient_context_count"],
        ),
        "",
    ]
    for case in evaluation["cases"]:
        lines.append("{} | {} | {} | hit@k={}".format(
            case["sample_id"],
            case["retrieval_status"],
            case["intent"],
            case["hit_at_k"] if case["hit_at_k"] is not None else "n/a",
        ))
        lines.append("  query: {}".format(case["query"]))
        lines.append("  expected: {}".format(", ".join(case["expected_chunk_ids"]) or "(none)"))
        lines.append("  retrieved: {}".format(
            ", ".join("{} ({:.4f}) {} / {}".format(
                item["chunk_id"], item["score"], item["source"], item["section"]
            ) for item in case["retrieved"]) or "(none)"
        ))
    return "\n".join(lines) + "\n"


def main():
    data_dir = default_data_dir()
    app_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect and evaluate Course Assistant retrieval")
    parser.add_argument("--data-dir", default=str(data_dir))
    parser.add_argument("--queries", default=str(data_dir / "sample_queries.txt"))
    parser.add_argument("--labeled-jsonl", help="evaluate query/expected chunk labels from JSONL")
    parser.add_argument("--output", default=str(app_dir / "output" / "retrieval.txt"))
    parser.add_argument("--evaluation-output", default=str(app_dir / "output" / "retrieval_evaluation.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--min-lexical-score", type=float, default=0.0)
    parser.add_argument("--no-bypass-unknown", action="store_true", help="send UNKNOWN samples through the retriever")
    parser.add_argument(
        "--use-gold-entities",
        action="store_true",
        help="use JSONL intent/entity labels as semantic hints (oracle/upper-bound mode)",
    )
    args = parser.parse_args()

    knowledge_base = KnowledgeBase(args.data_dir)
    retriever = Retriever(
        knowledge_base.documents,
        aliases=knowledge_base.aliases,
        top_k=args.top_k,
        min_score=args.min_score,
        min_lexical_score=args.min_lexical_score,
    )

    if args.labeled_jsonl:
        samples = read_labeled_jsonl(args.labeled_jsonl)
        evaluation = evaluate_samples(
            samples, retriever, args.top_k, use_gold_entities=args.use_gold_entities,
            bypass_unknown=not args.no_bypass_unknown,
        )
        output_path = Path(args.output)
        evaluation_path = Path(args.evaluation_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_evaluation_trace(evaluation), encoding="utf-8")
        evaluation_path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "Evaluated {} samples; Hit@{}={:.1%}; trace: {}; JSON: {}".format(
                evaluation["summary"]["sample_count"],
                args.top_k,
                evaluation["summary"]["hit_at_k"] or 0.0,
                output_path,
                evaluation_path,
            )
        )
        return

    reports = []
    for sample in read_queries(args.queries):
        if sample.get("intent") == "UNKNOWN" and not args.no_bypass_unknown:
            result = {
                "query": sample["query"],
                "chunks": [],
                "scores": [],
                "retrieval_status": "NO_RELEVANT_CONTEXT",
            }
        else:
            result = retriever.retrieve(sample["query"])
        reports.append("{}\n{}".format(sample["sample_id"], format_trace(result)))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(reports) + "\n", encoding="utf-8")
    print("Wrote retrieval results for {} queries to {}".format(len(reports), output_path))


if __name__ == "__main__":
    main()
