"""Unified NLP + real-answer evaluation entry point.

Run from python/:
    python -m nlp.app.unified_evaluation --provider mock
    python -m nlp.app.unified_evaluation --provider openai --require-real-model
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from nlp.app import answer_evaluation, evaluation


APP_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Run parser/semantic/retrieval evaluation and generated-answer evaluation"
    )
    parser.add_argument("--nlp-samples", default=str(APP_DIR / "output" / "samples_labeled.jsonl"))
    parser.add_argument("--answer-samples", default=str(APP_DIR / "input" / "answer_eval_smoke.jsonl"))
    parser.add_argument("--data-dir")
    parser.add_argument("--grammar")
    parser.add_argument("--provider", choices=("mock", "openai", "auto"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--require-real-model", action="store_true")
    parser.add_argument("--output-dir", default=str(APP_DIR / "output" / "unified_evaluation"))
    args = parser.parse_args()

    if args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    if args.judge and not args.judge_model:
        parser.error("--judge requires --judge-model")
    if args.require_real_model and args.provider != "openai":
        parser.error("--require-real-model requires --provider openai")

    nlp_rows = evaluation.subsample(
        evaluation.load_samples(args.nlp_samples), args.max_samples
    )
    nlp_report = evaluation.run_evaluation(
        nlp_rows, args.nlp_samples, data_dir=args.data_dir,
        grammar_path=args.grammar,
    )

    answer_rows = answer_evaluation.stratified_sample(
        answer_evaluation.load_dataset(args.answer_samples), args.max_samples
    )
    from nlp.app.assistant import CourseAssistant, create_llm_client

    generator = create_llm_client(provider=args.provider, model=args.model)
    if args.require_real_model and not getattr(generator, "use_api", False):
        parser.error("Real-model mode requires OPENAI_API_KEY and a real API provider")
    assistant = CourseAssistant(
        data_dir=args.data_dir, grammar_path=args.grammar, llm_client=generator
    )
    judge = answer_evaluation.OpenAICompatibleJudge(args.judge_model) if args.judge else None
    answer_results = answer_evaluation.evaluate_rows(assistant, answer_rows, judge=judge)

    if args.require_real_model:
        generated = [row for row in answer_results if row.get("model_call_count", 0) > 0]
        invalid = [row["sample_id"] for row in generated if row.get("model_execution") != "real_api"]
        if not generated or invalid:
            detail = ", ".join(invalid) if invalid else "no answer generation call was made"
            raise RuntimeError("Real-model requirement failed: {}".format(detail))

    answer_metrics = answer_evaluation.calculate_metrics(answer_results)
    answer_report = answer_evaluation.build_report(
        answer_rows, answer_results, answer_metrics, args.answer_samples,
        getattr(generator, "model_name", "unknown"),
        args.judge_model if judge else None,
        "full_dataset" if hasattr(assistant.kb, "entity_file") else "trace_fallback",
        len(getattr(assistant.kb, "documents", [])),
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "real_api_required" if args.require_real_model else args.provider,
        "nlp_pipeline": nlp_report,
        "answer_generation": answer_report,
        "interpretation": [
            "Parser, semantic and retrieval metrics are evaluated separately from generated-answer quality.",
            "Only REVIEWED rows with label_provenance=human_review enter answerability/retrieval headline metrics.",
            "Smoke rows are unreviewed diagnostics and must not be presented as gold evaluation.",
            "LLM-as-judge is diagnostic only and is skipped for mock, fallback or mixed-model outputs.",
            "A real-model run is verified only when successful API call telemetry is recorded and no fallback is present on generated rows.",
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "unified_evaluation.json"
    md_path = output_dir / "unified_evaluation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Unified Course Assistant Evaluation",
        "",
        "- Generated: {}".format(report["generated_at"]),
        "- Mode: {}".format(report["mode"]),
        "",
        "## Parser, semantic, retrieval and deterministic RAG",
        "",
        evaluation.render_markdown(nlp_report),
        "",
        "## Generated answer evaluation",
        "",
        answer_evaluation.render_markdown(answer_report),
        "",
        "## Interpretation",
        "",
    ]
    markdown.extend("- " + item for item in report["interpretation"])
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("Unified evaluation complete")
    print("JSON: {}".format(json_path))
    print("Markdown: {}".format(md_path))
    print("Answer model: {}".format(getattr(generator, "model_name", "unknown")))
    print("Successful API calls: {}".format(answer_metrics.get("successful_api_call_count", 0)))
    print("Fallback rows: {}".format(answer_metrics.get("fallback_count", 0)))


if __name__ == "__main__":
    main()
