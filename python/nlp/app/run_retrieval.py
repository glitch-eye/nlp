"""Generate a readable retrieval trace for the sample evaluation questions."""

import argparse
from pathlib import Path

from nlp.app.knowledge_base import KnowledgeBase, default_data_dir
from nlp.app.retriever import Retriever, format_trace


def read_queries(path):
    queries = []
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("=") or line.startswith("-") or "|" not in line:
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 2 and fields[0].startswith("Q"):
            queries.append((fields[0], fields[1]))
    return queries


def main():
    data_dir = default_data_dir()
    app_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect Course Assistant retrieval results")
    parser.add_argument("--data-dir", default=str(data_dir))
    parser.add_argument("--queries", default=str(data_dir / "sample_queries.txt"))
    parser.add_argument("--output", default=str(app_dir / "output" / "retrieval.txt"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    args = parser.parse_args()

    knowledge_base = KnowledgeBase(args.data_dir)
    retriever = Retriever(
        knowledge_base.documents,
        aliases=knowledge_base.aliases,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    reports = []
    for query_id, query in read_queries(args.queries):
        result = retriever.retrieve(query)
        reports.append("{}\n{}".format(query_id, format_trace(result)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(reports) + "\n", encoding="utf-8")
    print("Wrote retrieval results for {} queries to {}".format(len(reports), output_path))


if __name__ == "__main__":
    main()
