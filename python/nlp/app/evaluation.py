"""Core, reproducible evaluation for the Course Assistant NLP pipeline.

This module evaluates the existing single-intent mainline only:
  1) CFG parsing,
  2) semantic intent/entity extraction,
  3) retrieval against labelled expected chunks,
  4) end-to-end answerability and citation behaviour (offline mock by default).

It deliberately does not import or depend on the experimental multi-intent branch.
Run from python/: python -m nlp.app.evaluation
"""
import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from nlp.app.semantic_frame import frame_from_tree


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES = APP_DIR / "output" / "samples_labeled.jsonl"
VALID_ANSWERABILITY = {
    "EXPECTED_CONTEXT", "INSUFFICIENT_CONTEXT", "NO_RELEVANT_CONTEXT", "REVIEW"
}


def load_samples(path):
    """Load JSONL samples and fail with a useful line number on malformed data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Evaluation dataset not found: {}".format(path))
    samples = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON at {}:{}: {}".format(path, line_no, error))
        if not isinstance(row, dict) or not isinstance(row.get("query"), str):
            raise ValueError("Dataset row {} must be an object with a string 'query'".format(line_no))
        row = dict(row)
        row.setdefault("sample_id", "ROW{:05d}".format(line_no))
        row.setdefault("intent", "UNKNOWN")
        row.setdefault("entities", [])
        row.setdefault("expected_chunk_ids", [])
        row.setdefault("expected_answerability", "REVIEW")
        if not isinstance(row["entities"], list) or not isinstance(row["expected_chunk_ids"], list):
            raise ValueError("Dataset row {}: 'entities' and 'expected_chunk_ids' must be lists".format(line_no))
        if row["expected_answerability"] not in VALID_ANSWERABILITY:
            raise ValueError("Dataset row {} has unsupported expected_answerability={!r}".format(
                line_no, row["expected_answerability"]
            ))
        samples.append(row)
    if not samples:
        raise ValueError("Evaluation dataset is empty: {}".format(path))
    return samples


def subsample(rows, limit=None):
    """Evenly sample a dataset while retaining coverage across its original order."""
    if limit is None or limit <= 0 or len(rows) <= limit:
        return list(rows)
    step = len(rows) / float(limit)
    return [rows[int(index * step)] for index in range(limit)]


def _ratio(numerator, denominator):
    return numerator / float(denominator) if denominator else None


def _parse_query(parser, query):
    parse_multi = getattr(parser, "parse_multi", None)
    return parse_multi(query) if callable(parse_multi) else parser.parse(query)


def evaluate_parser(samples, parser):
    """Compare actual CFG acceptance with the dataset's expected parse_status."""
    confusion = Counter()
    cases = []
    exact_tree_checked = 0
    exact_tree_matches = 0
    for row in samples:
        tree = _parse_query(parser, row["query"])
        actual = "PARSED" if tree is not None else "UNPARSED"
        expected = row.get("parse_status")
        if expected in ("PARSED", "UNPARSED"):
            confusion[(expected, actual)] += 1
        expected_tree = row.get("parse_tree")
        tree_match = None
        if expected_tree and tree is not None:
            exact_tree_checked += 1
            tree_match = tree.to_bracketed() == expected_tree
            exact_tree_matches += int(tree_match)
        if expected != actual or tree_match is False:
            cases.append({
                "sample_id": row["sample_id"], "query": row["query"],
                "expected_parse_status": expected, "actual_parse_status": actual,
                "expected_tree": expected_tree, "actual_tree": tree.to_bracketed() if tree else None,
                "tree_match": tree_match,
            })
    evaluated = sum(confusion.values())
    correct = sum(count for (expected, actual), count in confusion.items() if expected == actual)
    return {
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": _ratio(correct, evaluated),
        "confusion": [
            {"expected": expected, "actual": actual, "count": count}
            for (expected, actual), count in sorted(confusion.items())
        ],
        "tree_exact_match_count": exact_tree_matches,
        "tree_exact_match_evaluated": exact_tree_checked,
        "tree_exact_match_rate": _ratio(exact_tree_matches, exact_tree_checked),
        "failures": cases,
    }


def _intent_set_from_row(row):
    if isinstance(row.get("intents"), list):
        return {value for value in row["intents"] if value}
    clauses = row.get("clauses")
    if isinstance(clauses, list):
        return {item.get("intent") for item in clauses if isinstance(item, dict) and item.get("intent")}
    intent = row.get("intent")
    return {intent} if intent and intent != "MULTI_INTENT" else set()


def evaluate_semantics(samples, parser, aliases=()):
    """Measure exact clause-intent-set accuracy and micro entity metrics."""
    intent_total = intent_correct = 0
    entity_tp = entity_fp = entity_fn = 0
    cases = []
    for row in samples:
        expected_intents = _intent_set_from_row(row)
        if not expected_intents:
            continue
        tree = _parse_query(parser, row["query"])
        frame = frame_from_tree(tree, row["query"], aliases) if tree is not None else {
            "intent": None, "entity_ids": [], "clauses": []
        }
        actual_clauses = frame.get("clauses") or [frame]
        actual_intents = {item.get("intent") for item in actual_clauses
                          if isinstance(item, dict) and item.get("intent")}
        if frame.get("intent") == "MULTI_INTENT":
            actual_intents.discard("MULTI_INTENT")
        intent_total += 1
        intent_correct += int(actual_intents == expected_intents)
        expected_entities = set(row.get("entities") or [])
        actual_entities = set(frame.get("entity_ids") or [])
        tp = len(expected_entities & actual_entities)
        fp = len(actual_entities - expected_entities)
        fn = len(expected_entities - actual_entities)
        entity_tp += tp
        entity_fp += fp
        entity_fn += fn
        if actual_intents != expected_intents or fp or fn:
            cases.append({
                "sample_id": row["sample_id"], "query": row["query"],
                "expected_intents": sorted(expected_intents),
                "actual_intents": sorted(actual_intents),
                "expected_entities": sorted(expected_entities),
                "actual_entities": sorted(actual_entities),
                "missing_entities": sorted(expected_entities - actual_entities),
                "unexpected_entities": sorted(actual_entities - expected_entities),
            })
    precision = _ratio(entity_tp, entity_tp + entity_fp)
    recall = _ratio(entity_tp, entity_tp + entity_fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall
          else (1.0 if precision == 1.0 and recall == 1.0 else None))
    return {
        "intent_evaluated": intent_total,
        "intent_correct": intent_correct,
        "intent_accuracy": _ratio(intent_correct, intent_total),
        "intent_match_definition": "exact set match across all clause intents",
        "entity_true_positive": entity_tp,
        "entity_false_positive": entity_fp,
        "entity_false_negative": entity_fn,
        "entity_precision_micro": precision,
        "entity_recall_micro": recall,
        "entity_f1_micro": f1,
        "failures": cases,
    }


def evaluate_retrieval(samples, retriever, top_k=3):
    """Use the existing retrieval evaluator in query-only mode (no gold-label leakage)."""
    from nlp.app.run_retrieval import evaluate_samples

    return evaluate_samples(
        samples, retriever, top_k,
        use_gold_entities=False,
        bypass_unknown=True,
    )


def evaluate_end_to_end(samples, assistant):
    """Evaluate the assistant with its configured client; CLI constructs an offline mock."""
    from nlp.app.rag_evaluation import evaluate_labeled, labeled_metrics

    results = evaluate_labeled(assistant, samples)
    metrics = labeled_metrics(results)
    confusion = [
        {"expected": expected, "actual": actual, "count": count}
        for (expected, actual), count in sorted(metrics["confusion"].items())
    ]
    return {
        "sample_count": metrics["n"],
        "answerable_count": metrics["with_ctx"],
        "retrieval_hit_count": metrics["hits"],
        "retrieval_hit_rate": _ratio(metrics["hits"], metrics["with_ctx"]),
        "answered_count": metrics["answered"],
        "answered_on_answerable_count": metrics["answered_ok"],
        "refusal_correct_count": metrics["refused_ok"],
        "should_refuse_count": metrics["should_refuse"],
        "refusal_accuracy": _ratio(metrics["refused_ok"], metrics["should_refuse"]),
        "answered_with_valid_citation_count": metrics["cited_ok"],
        "citation_validity_rate": _ratio(metrics["cited_ok"], metrics["answered"]),
        "needs_review_count": len(metrics["needs_review"]),
        "over_answered_count": len(metrics["over_answered"]),
        "status_confusion": confusion,
        "failures": [
            {
                "sample_id": row["id"], "query": row["query"],
                "expected": row["expected"], "actual_status": row["status"],
                "expected_chunks": sorted(row["expected_chunks"]),
                "retrieved_chunks": row["retrieved"],
            }
            for row in metrics["retrieval_misses"] + metrics["over_answered"]
        ],
    }


def _dataset_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_report(samples, sample_path, parser_metrics, semantic_metrics,
                 retrieval_metrics=None, e2e_metrics=None, config=None,
                 rag_evidence=None):
    """Create a JSON-safe report with provenance and measured metrics."""
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "dataset": {
            "path": str(Path(sample_path)),
            "sha256": _dataset_digest(sample_path),
            "sample_count": len(samples),
            "label_status_counts": dict(Counter(row.get("label_status", "UNSPECIFIED") for row in samples)),
            "answerability_counts": dict(Counter(row["expected_answerability"] for row in samples)),
        },
        "config": config or {},
        "metrics": {
            "parser": parser_metrics,
            "semantic": semantic_metrics,
            "retrieval": retrieval_metrics,
            "end_to_end": e2e_metrics,
        },
        "rag_evidence": rag_evidence or [],
        "interpretation_notes": [
            "Metrics measure the supplied dataset; they are not proof of production quality.",
            "AUTO_LABEL_REQUIRES_REVIEW labels must be manually audited before academic claims.",
            "Retrieval evaluation uses the original query without gold intent/entity hints.",
            "End-to-end evaluation uses the offline mock when launched by this CLI; it is not a real-LLM quality score.",
        ],
    }


def render_markdown(report):
    """Render a concise human-readable summary; detailed per-case data stays in JSON."""
    metrics = report["metrics"]
    lines = [
        "# Course Assistant Evaluation",
        "",
        "- Generated: {}".format(report["generated_at"]),
        "- Dataset: {}".format(report["dataset"]["path"]),
        "- Samples: {}".format(report["dataset"]["sample_count"]),
        "- SHA-256: `{}`".format(report["dataset"]["sha256"]),
        "- Knowledge base: {}".format(report["config"].get("knowledge_base_mode", "unspecified")),
        "- KB documents: {}".format(report["config"].get("knowledge_base_document_count", "n/a")),
        "",
        "## Parser",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Parse-status accuracy | {} |".format(_fmt(metrics["parser"]["accuracy"])),
        "| Exact parse-tree match | {} ({}/{}) |".format(
            _fmt(metrics["parser"]["tree_exact_match_rate"]),
            metrics["parser"]["tree_exact_match_count"],
            metrics["parser"]["tree_exact_match_evaluated"],
        ),
        "",
        "## Semantic interpretation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Intent accuracy | {} |".format(_fmt(metrics["semantic"]["intent_accuracy"])),
        "| Entity precision (micro) | {} |".format(_fmt(metrics["semantic"]["entity_precision_micro"])),
        "| Entity recall (micro) | {} |".format(_fmt(metrics["semantic"]["entity_recall_micro"])),
        "| Entity F1 (micro) | {} |".format(_fmt(metrics["semantic"]["entity_f1_micro"])),
        "",
    ]
    retrieval = metrics.get("retrieval")
    if retrieval:
        summary = retrieval["summary"]
        lines.extend([
            "## Retrieval",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| Hit@{} | {} ({}/{}) |".format(
                summary["top_k"], _fmt(summary["hit_at_k"]),
                summary["hit_at_k_count"], summary["expected_context_count"],
            ),
            "| UNKNOWN rejection rate | {} |".format(_fmt(summary["unknown_rejection_rate"])),
            "| Retrieved for insufficient-context cases | {}/{} |".format(
                summary["insufficient_context_retrieved_count"], summary["insufficient_context_count"]
            ),
            "",
        ])
    e2e = metrics.get("end_to_end")
    if e2e:
        lines.extend([
            "## End-to-end",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| Correct refusal rate | {} |".format(_fmt(e2e["refusal_accuracy"])),
            "| Valid citation rate among ANSWERED | {} |".format(_fmt(e2e["citation_validity_rate"])),
            "| NEEDS_REVIEW | {} |".format(e2e["needs_review_count"]),
            "| Over-answered insufficient-context cases | {} |".format(e2e["over_answered_count"]),
            "",
        ])
    lines.extend(["## Representative failures", ""])
    failure_groups = [
        ("Parser", metrics["parser"].get("failures", [])),
        ("Semantic", metrics["semantic"].get("failures", [])),
        ("End-to-end", (e2e or {}).get("failures", [])),
    ]
    for stage, failures in failure_groups:
        lines.append("### {} ({} cases; showing at most 10)".format(stage, len(failures)))
        for case in failures[:10]:
            lines.append("- {}: {}".format(case.get("sample_id", "unknown"), case.get("query", "")))
            if stage == "Parser":
                lines.append("  - expected={}, actual={}".format(
                    case.get("expected_parse_status"), case.get("actual_parse_status")))
            elif stage == "Semantic":
                lines.append("  - expected intent={}, actual intent={}; missing entities={}, unexpected entities={}".format(
                    case.get("expected_intent"), case.get("actual_intent"),
                    case.get("missing_entities"), case.get("unexpected_entities")))
            else:
                lines.append("  - expected={}, actual status={}, expected chunks={}, retrieved={}".format(
                    case.get("expected"), case.get("actual_status"),
                    case.get("expected_chunks"), case.get("retrieved_chunks")))
        if not failures:
            lines.append("- No failures recorded.")
        lines.append("")
    lines.extend([
        "## Strengths observed in the implementation",
        "",
        "- The pipeline separates parsing, semantic interpretation, retrieval and answer generation, so failures can be inspected by stage.",
        "- The aggregate evaluation uses the offline mock by default and records dataset/runtime provenance for repeatability.",
        "- The RAG evidence section exercises grounded answers, no-information cases and scripted Answer Guard probes.",
        "",
        "## Limitations and error-analysis guidance",
        "",
        "- CFG coverage is limited to patterns represented in the grammar; unseen paraphrases may be UNPARSED.",
        "- TF-IDF and rule-based semantic signals can miss relevant chunks when wording differs substantially.",
        "- Mock results do not measure the quality of a real LLM; citation matching also does not prove full semantic entailment.",
        "- AUTO_LABEL_REQUIRES_REVIEW labels and expected chunk IDs must be audited before interpreting scores as ground truth.",
        "- If the external course dataset is unavailable, trace fallback data can make retrieval metrics unrepresentative.",
        "- Inspect parser failures as grammar/tokenization gaps, semantic failures as intent/entity mapping gaps, and retrieval/end-to-end failures as KB coverage, ranking, answer-guard or label-quality issues.",
        "",
        "## Interpretation cautions",
        "",
    ])
    lines.extend("- " + note for note in report["interpretation_notes"])
    lines.append("")
    return "\n".join(lines)


def _fmt(value):
    return "n/a" if value is None else "{:.1%}".format(value)


def run_evaluation(samples, sample_path, data_dir=None, grammar_path=None,
                   top_k=3, min_score=0.08, include_retrieval=True,
                   include_end_to_end=True):
    """Run the mainline evaluation stages without contacting a paid provider."""
    from nlp.app.assistant import CourseAssistant, create_llm_client

    # Reuse CourseAssistant's KB loading/fallback policy so the evaluator works
    # both with the full external dataset and with the repository trace fallback.
    assistant = CourseAssistant(
        data_dir=data_dir, grammar_path=grammar_path,
        top_k=top_k, llm_client=create_llm_client(provider="mock"),
    )
    kb = assistant.kb
    parser = assistant.parser
    retriever = assistant.retriever
    retriever.top_k = top_k
    retriever.min_score = min_score

    parser_metrics = evaluate_parser(samples, parser)
    semantic_metrics = evaluate_semantics(samples, parser, kb.aliases)
    retrieval_metrics = evaluate_retrieval(samples, retriever, top_k) if include_retrieval else None
    e2e_metrics = evaluate_end_to_end(samples, assistant) if include_end_to_end else None
    rag_evidence = None
    if include_end_to_end:
        from nlp.app.rag_evaluation import build_rag_sections
        rag_evidence = build_rag_sections(assistant, samples_path=sample_path, max_samples=len(samples))

    return build_report(
        samples, sample_path, parser_metrics, semantic_metrics,
        retrieval_metrics, e2e_metrics,
        config={
            "top_k": top_k, "min_score": min_score,
            "retrieval_uses_gold_entities": False,
            "llm_provider": "mock" if include_end_to_end else None,
            "knowledge_base_mode": "full_dataset" if hasattr(kb, "entity_file") else "trace_fallback",
            "knowledge_base_document_count": len(getattr(kb, "documents", [])),
            "knowledge_base_entity_count": len(getattr(kb, "entities", {})) if hasattr(kb, "entities") else None,
            "stages": {
                "parser": True, "semantic": True,
                "retrieval": include_retrieval,
                "end_to_end": include_end_to_end,
            },
        },
        rag_evidence=rag_evidence,
    )


def main():
    app_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Evaluate parser, semantics, retrieval and end-to-end Course Assistant")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="labelled JSONL dataset")
    ap.add_argument("--data-dir", help="optional full sample-data directory; otherwise use CourseAssistant fallback policy")
    ap.add_argument("--grammar", help="optional grammar file override")
    ap.add_argument("--output-dir", default=str(app_dir / "output"), help="directory for evaluation.txt and evaluation_report.json")
    ap.add_argument("--max-samples", type=int, help="evenly subsample N rows; default evaluates all")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.08)
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    if args.top_k < 1:
        ap.error("--top-k must be >= 1")
    if args.min_score < 0:
        ap.error("--min-score must be >= 0")
    samples = load_samples(args.samples)
    samples = subsample(samples, args.max_samples)
    report = run_evaluation(
        samples, args.samples, data_dir=args.data_dir,
        grammar_path=args.grammar, top_k=args.top_k, min_score=args.min_score,
        include_retrieval=not args.skip_retrieval,
        include_end_to_end=not args.skip_e2e,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_report.json"
    text_path = output_dir / "evaluation.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_report = render_markdown(report)
    if report.get("rag_evidence"):
        text_report += "\n\n## RAG evaluation evidence\n\n" + "\n".join(report["rag_evidence"])
    text_path.write_text(text_report + "\n", encoding="utf-8")
    print("Evaluation complete: {} samples".format(len(samples)))
    print("Report: {}".format(text_path))
    print("Details: {}".format(json_path))
    print(text_report)


if __name__ == "__main__":
    main()
