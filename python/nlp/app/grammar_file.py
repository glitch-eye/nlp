"""Reader for the text CFG format used by ``output/grammar.txt``."""

from pathlib import Path
import re


RULE_RE = re.compile(r"^\s*<([^<>]+)>\s*::=\s*(.*?)\s*$")
SYMBOL_RE = re.compile(r'<([^<>]+)>|"((?:\\.|[^"\\])*)"|([^\s]+)')


def _symbols(text, line_number):
    """Parse nonterminals and quoted terminals from one alternative."""
    result = []
    position = 0
    for match in SYMBOL_RE.finditer(text):
        if text[position:match.start()].strip():
            raise ValueError("Invalid grammar syntax on line {}".format(line_number))
        if match.group(1) is not None:
            result.append(match.group(1))
        elif match.group(2) is not None:
            result.append(match.group(2).replace(r'\"', '"').replace(r'\\', '\\'))
        else:
            raise ValueError("Terminals must be quoted on line {}".format(line_number))
        position = match.end()
    if text[position:].strip():
        raise ValueError("Invalid grammar syntax on line {}".format(line_number))
    return result


def _alternatives(text):
    """Split alternatives on unquoted pipes."""
    chunks = []
    quoted = False
    escaped = False
    start = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "|" and not quoted:
            chunks.append(text[start:index].strip())
            start = index + 1
    chunks.append(text[start:].strip())
    return chunks


def load_grammar(path):
    """Return ``{nonterminal: [production, ...]}`` from a multiline CFG file."""
    path = Path(path)
    productions = {}
    current = None

    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            rule = RULE_RE.match(raw_line)
            if rule:
                current = rule.group(1).strip()
                productions.setdefault(current, [])
                alternative = rule.group(2)
                productions[current].extend(
                    _symbols(chunk, line_number) for chunk in _alternatives(alternative)
                )
                continue

            if line.startswith("|"):
                if current is None:
                    raise ValueError("Alternative without a rule on line {}".format(line_number))
                alternative = line[1:].strip()
                productions[current].extend(
                    _symbols(chunk, line_number) for chunk in _alternatives(alternative)
                )
                continue

            raise ValueError("Unrecognized grammar syntax on line {}".format(line_number))

    if not productions:
        raise ValueError("Grammar file contains no production rules: {}".format(path))
    if "S" not in productions:
        raise ValueError("Grammar file has no <S> start rule: {}".format(path))
    return productions
