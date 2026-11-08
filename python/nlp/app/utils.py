"""Generate Course Assistant questions from the configured CFG."""

import argparse
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = APP_DIR.parents[2]  # hcmut/, so `iaslab.*` imports resolve.
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from nlp.app.course_grammar import CourseGrammar
from nlp.app.sentence_generator import generate_to_file


def main():
    parser = argparse.ArgumentParser(description="Generate Course Assistant sample questions")
    parser.add_argument(
        "--output",
        default=str(APP_DIR / "output" / "samples.txt"),
        help="output text file (default: app/output/samples.txt)",
    )
    parser.add_argument("--max-sentences", type=int, default=10000)
    args = parser.parse_args()

    grammar = CourseGrammar("vi")
    sentences = generate_to_file(
        grammar,
        output_path=args.output,
        max_sentences=args.max_sentences,
    )
    print("Generated {} sentence(s) in {}".format(len(sentences), args.output))


if __name__ == "__main__":
    main()
