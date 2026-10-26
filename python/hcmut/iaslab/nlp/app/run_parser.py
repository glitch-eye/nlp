"""Command line entry point for the Part I parser."""

import argparse
from pathlib import Path

from iaslab.nlp.app.parser import CourseParser


def main():
    app_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Parse Course Assistant sentences")
    parser.add_argument("--input", default=str(app_dir / "input" / "sentences.txt"))
    parser.add_argument("--output", default=str(app_dir / "output" / "parse-results.txt"))
    parser.add_argument("--grammar", default=str(app_dir / "output" / "grammar.txt"))
    args = parser.parse_args()
    results = CourseParser(args.grammar).parse_file(args.input, args.output)
    print("Parsed {} sentence(s) into {}".format(len(results), args.output))


if __name__ == "__main__":
    main()
