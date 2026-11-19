"""Evaluate generated answers from the real Course Assistant pipeline.

This is deliberately separate from parser/grammar evaluation. It measures:
- deterministic answerability/refusal and retrieval behaviour against reviewed labels;
- citation/guard outcomes recorded by the assistant;
- optional model-as-judge scores against a human-authored reference answer and the
  exact retrieved context shown to the generator.

A judge score is diagnostic, not ground truth. Rows without reference_answer are
not scored for correctness/completeness. Do not treat synthetic or auto-labelled
rows as a human-reviewed gold set.
Run from python/:
    python -m nlp.app.answer_evaluation --samples nlp/app/input/test_labeled.jsonl
    python -m nlp.app.answer_evaluation --samples nlp/app/input/answer_eval.jsonl \
        --provider openai --model gpt-4o-mini --judge --judge-model <separate-judge-model>
"""

import argparse
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES = APP_DIR / "input" / "test_labeled.jsonl"
ANSWERABLE = "EXPECTED_CONTEXT"
REFUSAL_LABELS = {"INSUFFICIENT_CONTEXT", "NO_RELEVANT_CONTEXT"}
REFUSAL_STATUSES = {"INSUFFICIENT_CONTEXT", "NO_RELEVANT_CONTEXT"}
JUDGE_SCORE_KEYS = ("correctness", "faithfulness", "relevancy", "completeness")


def load_dataset(path):
    """Load answer-evaluation JSONL and validate fields without inventing references."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Evaluation dataset not found: {}".format(path))
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON at {}:{}: {}".format(path, line_no, exc))
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError("Row {} must be an object with a non-empty query".format(line_no))
        row = dict(row)
        row.setdefault("sample_id", "ANSWER{:05d}".format(line_no))
        row.setdefault("expected_answerability", "REVIEW")
        row.setdefault("expected_chunk_ids", [])
        if row["expected_answerability"] not in {
            ANSWERABLE, "INSUFFICIENT_CONTEXT", "NO_RELEVANT_CONTEXT", "REVIEW"
        }:
            raise ValueError("Unsupported expected_answerability at row {}".format(line_no))
        if not isinstance(row["expected_chunk_ids"], list):
            raise ValueError("expected_chunk_ids must be a list at row {}".format(line_no))
        if row.get("reference_answer") is not None and not isinstance(row["reference_answer"], str):
            raise ValueError("reference_answer must be a string at row {}".format(line_no))
        if row.get("reference_points") is not None and not isinstance(row["reference_points"], list):
            raise ValueError("reference_points must be a list at row {}".format(line_no))
        rows.append(row)
    if not rows:
        raise ValueError("Evaluation dataset is empty: {}".format(path))
    return rows


def stratified_sample(rows, limit=None):
    """Deterministic round-robin sample by answerability so a cap keeps class coverage."""
    if limit is None or limit <= 0 or len(rows) <= limit:
        return list(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("expected_answerability", "REVIEW")].append(row)
    keys = sorted(groups)
    selected = []
    index = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if index < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][index])
                added = True
        if not added:
            break
        index += 1
    return selected


def parse_judge_json(raw):
    """Parse strict JSON or a JSON object wrapped in a Markdown code fence."""
    text = (raw or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Judge response did not contain a JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("Judge returned malformed JSON: {}".format(exc))
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Judge response is missing the scores object")
    normalized = {}
    for key in JUDGE_SCORE_KEYS:
        score = scores.get(key)
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5):
            raise ValueError("Judge score {} must be null or between 1 and 5".format(key))
        normalized[key] = float(score) if score is not None else None
    hallucination = value.get("hallucination")
    if hallucination not in (True, False, None):
        raise ValueError("Judge hallucination must be true, false, or null")
    return {
        "scores": normalized,
        "hallucination": hallucination,
        "unsupported_claims": value.get("unsupported_claims") or [],
        "missing_points": value.get("missing_points") or [],
        "reason": str(value.get("reason") or ""),
    }


class OpenAICompatibleJudge:
    """Separate, opt-in judge client; never silently falls back to the answer model."""

    def __init__(self, model, api_key=None, base_url=None, timeout=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = float(timeout or os.environ.get("LLM_TIMEOUT", 30))
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when --judge is enabled")

    @staticmethod
    def build_prompt(row, result):
        chunks = result.get("chunks") or []
        context = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "section": chunk.get("section"),
                "text": chunk.get("text", ""),
            }
            for chunk in chunks
        ]
        reference_is_reviewed = row.get("reference_status") == "HUMAN_REVIEWED"
        payload = {
            "question": row["query"],
            "reference_answer": row.get("reference_answer") if reference_is_reviewed else None,
            "reference_points": row.get("reference_points") if reference_is_reviewed else None,
            "expected_answerability": row.get("expected_answerability"),
            "retrieved_context": context,
            "generated_answer": result.get("answer", ""),
            "system_status": result.get("status"),
        }
        return (
            "Evaluate the generated answer for a Vietnamese university course assistant. "
            "Treat retrieved_context as the only evidence for faithfulness. "
            "Use reference_answer/reference_points as the answer key only when supplied. "
            "Do not reward a plausible claim that is unsupported by context. "
            "If no reference answer or points are supplied, set correctness and completeness to null. "
            "If the system refused and the expected_answerability is not EXPECTED_CONTEXT, "
            "judge whether refusal is appropriate; scores may be null when not applicable. "
            "Score applicable dimensions from 1 (poor) to 5 (excellent). "
            "hallucination=true only when the answer asserts a material claim not supported by context. "
            "Return ONLY JSON with this schema: "
            '{"scores":{"correctness":1-5|null,"faithfulness":1-5|null,'
            '"relevancy":1-5|null,"completeness":1-5|null},'
            '"hallucination":true|false|null,"unsupported_claims":[],"missing_points":[],"reason":"short"}'
            "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
        )

    def evaluate(self, row, result):
        prompt = self.build_prompt(row, result)
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict evaluator, not an answer generator. Never infer facts beyond the supplied evidence.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            raw = response_body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError("Judge API failed: {}".format(type(exc).__name__))
        return parse_judge_json(raw)


def _ratio(numerator, denominator):
    return numerator / float(denominator) if denominator else None


def _is_fallback(model_name):
    return "[FALLBACK:" in str(model_name or "")


def evaluate_rows(assistant, rows, judge=None):
    """Run the actual assistant and optionally judge each generated answer."""
    results = []
    for row in rows:
        started = time.perf_counter()
        output = assistant.answer(row["query"])
        elapsed_seconds = time.perf_counter() - started
        model_name = getattr(assistant.llm_client, "model_name", "unknown")
        last_error = getattr(assistant.llm_client, "last_error", None)
        record = {
            "sample_id": row.get("sample_id"),
            "query": row["query"],
            "expected_answerability": row.get("expected_answerability", "REVIEW"),
            "label_status": row.get("label_status", "UNSPECIFIED"),
            "label_provenance": row.get("label_provenance", "UNSPECIFIED"),
            "expected_chunk_ids": list(row.get("expected_chunk_ids") or []),
            "retrieved_chunk_ids": list(output.get("retrieved_chunk_ids") or []),
            "parse_status": output.get("parse_status"),
            "retrieval_status": output.get("retrieval_status"),
            "actual_status": output.get("status"),
            "answer": output.get("answer", ""),
            "sources": output.get("sources") or [],
            "citation_check": output.get("citation_check") or {},
            "model_name": model_name,
            "model_fallback": _is_fallback(model_name),
            "model_error": last_error,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "reference_status": row.get("reference_status", "UNVERIFIED"),
            "judge": None,
            "judge_error": None,
        }
        if judge is not None:
            if record["model_fallback"] or "mock" in str(model_name).lower():
                record["judge_error"] = "SKIPPED: generator output is mock or fallback, not a real-model answer"
            else:
                try:
                    record["judge"] = judge.evaluate(row, output)
                except Exception as exc:
                    record["judge_error"] = str(exc)
        results.append(record)
    return results


def calculate_metrics(results):
    """Calculate deterministic metrics; REVIEW rows are excluded from headline rates."""
    # Only explicit human-review provenance is eligible for headline gold metrics.
    labelled = [r for r in results if r["expected_answerability"] != "REVIEW"
                and r.get("label_status") == "REVIEWED"
                and r.get("label_provenance") == "human_review"]
    expected_answerable = [r for r in labelled if r["expected_answerability"] == ANSWERABLE]
    expected_refusal = [r for r in labelled if r["expected_answerability"] in REFUSAL_LABELS]
    answered = [r for r in results if r["actual_status"] == "ANSWERED"]
    refused = [r for r in results if r["actual_status"] in REFUSAL_STATUSES]
    needs_review = [r for r in results if r["actual_status"] == "NEEDS_REVIEW"]

    expected_total = 0
    retrieved_total = 0
    true_positive = 0
    hit = 0
    for row in expected_answerable:
        expected = set(row["expected_chunk_ids"])
        retrieved = set(row["retrieved_chunk_ids"])
        expected_total += len(expected)
        retrieved_total += len(retrieved)
        true_positive += len(expected & retrieved)
        hit += int(bool(expected & retrieved))

    citation_ok = 0
    citation_checked = 0
    for row in answered:
        check = row.get("citation_check") or {}
        valid = check.get("valid_citations") or []
        invalid = check.get("invalid_citations") or []
        if valid or invalid:
            citation_checked += 1
            citation_ok += int(bool(valid) and not invalid)

    judged = [r["judge"] for r in results if r.get("judge")]
    reference_reviewed_count = sum(r.get("reference_status") == "HUMAN_REVIEWED" for r in results)
    score_means = {}
    for key in JUDGE_SCORE_KEYS:
        values = [j["scores"].get(key) for j in judged if j.get("scores", {}).get(key) is not None]
        score_means[key] = {
            "mean": (sum(values) / len(values)) if values else None,
            "n": len(values),
        }
    hallucination_checked = [j for j in judged if j.get("hallucination") is not None]
    return {
        "sample_count": len(results),
        "labelled_count": len(labelled),
        "review_label_count": len(results) - len(labelled),
        "human_reviewed_label_count": len(labelled),
        "unverified_label_count": len(results) - len(labelled),
        "human_reviewed_reference_count": reference_reviewed_count,
        "answerable_count": len(expected_answerable),
        "expected_refusal_count": len(expected_refusal),
        "answered_count": len(answered),
        "refused_count": len(refused),
        "needs_review_count": len(needs_review),
        "answer_rate_on_answerable": _ratio(
            sum(r["actual_status"] == "ANSWERED" for r in expected_answerable), len(expected_answerable)
        ),
        "correct_refusal_rate": _ratio(
            sum(r["actual_status"] in REFUSAL_STATUSES for r in expected_refusal), len(expected_refusal)
        ),
        "over_answered_count": sum(r["actual_status"] == "ANSWERED" for r in expected_refusal),
        "retrieval_hit_at_k": _ratio(hit, len(expected_answerable)),
        "retrieval_micro_precision": _ratio(true_positive, retrieved_total),
        "retrieval_micro_recall": _ratio(true_positive, expected_total),
        "citation_validity_rate": _ratio(citation_ok, citation_checked),
        "citation_checked_count": citation_checked,
        "judge_score_means": score_means,
        "judge_hallucination_rate": _ratio(
            sum(j["hallucination"] is True for j in hallucination_checked), len(hallucination_checked)
        ),
        "judge_evaluated_count": len(judged),
        "judge_error_count": sum(bool(r.get("judge_error")) and not r["judge_error"].startswith("SKIPPED:") for r in results),
        "judge_skipped_count": sum(bool(r.get("judge_error")) and r["judge_error"].startswith("SKIPPED:") for r in results),
        "fallback_count": sum(r["model_fallback"] for r in results),
        "mock_answer_count": sum("mock" in str(r.get("model_name", "")).lower() for r in results),
        "real_model_answer_count": sum(
            not r["model_fallback"] and "mock" not in str(r.get("model_name", "")).lower()
            for r in results
        ),
        "status_confusion": _status_confusion(labelled),
    }


def _status_confusion(rows):
    counts = defaultdict(int)
    for row in rows:
        counts["{} -> {}".format(row["expected_answerability"], row["actual_status"])] += 1
    return [{"pair": key, "count": counts[key]} for key in sorted(counts)]


def build_report(rows, results, metrics, dataset_path, generator_model, judge_model=None,
                 knowledge_base_mode="unknown", knowledge_base_document_count=None):
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sample_count": len(rows),
            "label_status_counts": _count(row.get("label_status", "UNSPECIFIED") for row in rows),
            "provenance_counts": _count(row.get("label_provenance", "UNSPECIFIED") for row in rows),
            "reference_answer_count": sum(bool(row.get("reference_answer")) for row in rows),
            "human_reviewed_reference_count": sum(row.get("reference_status") == "HUMAN_REVIEWED" for row in rows),
            "reference_points_count": sum(bool(row.get("reference_points")) for row in rows),
        },
        "models": {"generator": generator_model, "judge": judge_model},
        "knowledge_base": {
            "mode": knowledge_base_mode,
            "document_count": knowledge_base_document_count,
        },
        "metrics": metrics,
        "results": results,
        "limitations": [
            "A trace-fallback Knowledge Base is not equivalent to the full course dataset; do not report its retrieval scores as representative.",
            "Citation validity only checks whether a source label was among the chunks sent to the model; it does not prove semantic support.",
            "The answer guard's number/date checks do not detect every unsupported natural-language claim.",
            "LLM-as-judge is a secondary signal, not ground truth; calibrate it against human-reviewed examples.",
            "Correctness/completeness are scored against references only when reference_status is HUMAN_REVIEWED; unverified reference text is withheld from the judge.",
            "Auto-labelled and data-source-reviewed rows are not equivalent to independently human-reviewed answer gold.",
            "A fallback answer is not an output from the requested real model and must be excluded from real-model quality claims.",
        ],
    }


def _count(values):
    counts = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def render_markdown(report):
    m = report["metrics"]
    models = report["models"]
    lines = [
        "# LLM Answer Evaluation",
        "",
        "- Generated: {}".format(report["generated_at"]),
        "- Dataset: {}".format(report["dataset"]["path"]),
        "- Samples: {}".format(report["dataset"]["sample_count"]),
        "- Generator: {}".format(models.get("generator")),
        "- Judge: {}".format(models.get("judge") or "disabled"),
        "- Rows with reference answers: {}".format(report["dataset"]["reference_answer_count"]),
        "- Knowledge base: {} ({} documents)".format(
            report["knowledge_base"]["mode"], report["knowledge_base"]["document_count"]
        ),
        "",
        "## Deterministic metrics",
        "",
        "> Headline answerability/retrieval rates use only rows with label_status=REVIEWED and label_provenance=human_review. Other rows are diagnostic, not gold.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        "| Human-reviewed gold rows | {} |".format(m["human_reviewed_label_count"]),
        "| Answer rate on answerable questions (human-reviewed only) | {} |".format(_fmt(m["answer_rate_on_answerable"])),
        "| Correct refusal rate | {} |".format(_fmt(m["correct_refusal_rate"])),
        "| Over-answered insufficient/no-context cases | {} |".format(m["over_answered_count"]),
        "| Retrieval Hit@k | {} |".format(_fmt(m["retrieval_hit_at_k"])),
        "| Retrieval micro precision / recall | {} / {} |".format(_fmt(m["retrieval_micro_precision"]), _fmt(m["retrieval_micro_recall"])),
        "| Citation validity (checked ANSWERED only) | {} |".format(_fmt(m["citation_validity_rate"])),
        "| NEEDS_REVIEW | {} |".format(m["needs_review_count"]),
        "| Real-model answer rows | {} |".format(m["real_model_answer_count"]),
        "| Mock answer rows | {} |".format(m["mock_answer_count"]),
        "| Model fallback rows | {} |".format(m["fallback_count"]),
        "",
        "## Model-as-judge (diagnostic only)",
        "",
        "| Dimension | Mean score (1–5) | N |",
        "|---|---:|---:|",
    ]
    for key in JUDGE_SCORE_KEYS:
        item = m["judge_score_means"][key]
        lines.append("| {} | {} | {} |".format(key.title(), _score(item["mean"]), item["n"]))
    lines.extend([
        "| Hallucination rate (judge-flagged) | {} |".format(_fmt(m["judge_hallucination_rate"])),
        "| Human-reviewed references | {} |".format(m["human_reviewed_reference_count"]),
        "| Judge evaluated / errors / skipped | {} / {} / {} |".format(m["judge_evaluated_count"], m["judge_error_count"], m["judge_skipped_count"]),
        "",
        "## Status confusion",
        "",
        "| Expected → actual | Count |",
        "|---|---:|",
    ])
    for item in m["status_confusion"]:
        lines.append("| {} | {} |".format(item["pair"], item["count"]))
    lines.extend(["", "## Limitations"])
    lines.extend("- " + note for note in report["limitations"])
    lines.extend(["", "## Per-sample details", ""])
    for row in report["results"]:
        lines.append("### {} — {}".format(row["sample_id"], row["query"]))
        lines.append("- Expected: {}; actual: {}".format(row["expected_answerability"], row["actual_status"]))
        lines.append("- Model: {}{}".format(row["model_name"], " (FALLBACK)" if row["model_fallback"] else ""))
        lines.append("- Retrieved: {}".format(", ".join(row["retrieved_chunk_ids"]) or "(none)"))
        lines.append("- Answer: {}".format((row["answer"] or "").replace("\n", " ")))
        if row.get("judge"):
            lines.append("- Judge: {}".format(json.dumps(row["judge"], ensure_ascii=False)))
        if row.get("judge_error"):
            lines.append("- Judge error: {}".format(row["judge_error"]))
        lines.append("")
    return "\n".join(lines)


def _fmt(value):
    return "n/a" if value is None else "{:.1%}".format(value)


def _score(value):
    return "n/a" if value is None else "{:.2f}".format(value)


def main():
    from nlp.app.assistant import CourseAssistant, create_llm_client

    ap = argparse.ArgumentParser(description="Evaluate real Course Assistant generated answers")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="JSONL with query and reviewed expectations")
    ap.add_argument("--data-dir", help="course knowledge-base directory")
    ap.add_argument("--grammar", help="optional grammar override")
    ap.add_argument("--provider", choices=("mock", "openai", "auto"), default=None)
    ap.add_argument("--model", help="generator model id")
    ap.add_argument("--max-samples", type=int, default=50, help="maximum rows per run (default: 50)")
    ap.add_argument("--judge", action="store_true", help="enable a separate model-as-judge API call per row")
    ap.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL"), help="judge model id; prefer a different model from the generator")
    ap.add_argument("--output-dir", default=str(APP_DIR / "output" / "answer_evaluation"))
    args = ap.parse_args()

    if args.judge and not args.judge_model:
        ap.error("--judge requires --judge-model or EVAL_JUDGE_MODEL")
    rows = stratified_sample(load_dataset(args.samples), args.max_samples)
    generator = create_llm_client(provider=args.provider, model=args.model)
    assistant = CourseAssistant(data_dir=args.data_dir, grammar_path=args.grammar, llm_client=generator)
    judge = OpenAICompatibleJudge(args.judge_model) if args.judge else None
    results = evaluate_rows(assistant, rows, judge=judge)
    metrics = calculate_metrics(results)
    report = build_report(
        rows, results, metrics, args.samples,
        getattr(generator, "model_name", "unknown"),
        args.judge_model if judge else None,
        "full_dataset" if hasattr(assistant.kb, "entity_file") else "trace_fallback",
        len(getattr(assistant.kb, "documents", [])),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "answer_evaluation.json"
    md_path = output_dir / "answer_evaluation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    print("Evaluated {} samples".format(len(rows)))
    print("Markdown report: {}".format(md_path))
    print("JSON details: {}".format(json_path))
    print(markdown)


if __name__ == "__main__":
    main()
