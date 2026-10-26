from pathlib import Path
import unicodedata
from functools import lru_cache

from iaslab.nlp.app.grammar_file import load_grammar


class ParseNode:
    """Nút biểu diễn cây cú pháp dạng ngoặc đơn."""

    def __init__(self, label, children=()):
        self.label = label
        self.children = tuple(children)

    def to_bracketed(self):
        if not self.children:
            return f"({self.label})"
        return f"({self.label} {' '.join(child.to_bracketed() for child in self.children)})"


class CourseParser:
    """Bộ phân tích cú pháp Earley Parser cho văn phạm Course Assistant."""

    def __init__(self, grammar_path=None):
        app_dir = Path(__file__).resolve().parent
        self.grammar_path = Path(
            grammar_path or app_dir / "output" / "grammar.txt"
        )
        raw_rules = load_grammar(self.grammar_path)
        names = set(raw_rules)
        self.rules = {
            name: tuple(
                tuple(("N", sym) if sym in names else ("T", sym) for sym in prod)
                for prod in prods
            )
            for name, prods in raw_rules.items()
        }
        self._parse_cached = lru_cache(maxsize=50000)(self._parse_uncached)

    @staticmethod
    def _skip_spaces(text, pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    def _parse_uncached(self, sentence):
        text = unicodedata.normalize("NFC", sentence.strip())
        end = len(text)
        start, start_prod = "__START__", (("N", "S"),)

        chart = [set() for _ in range(end + 1)]
        agenda = [[] for _ in range(end + 1)]

        def add(pos, lhs, prod, dot, origin, children):
            pos = self._skip_spaces(text, pos)
            state = (lhs, prod, dot, origin, children)
            if state not in chart[pos]:
                chart[pos].add(state)
                agenda[pos].append(state)

        add(0, start, start_prod, 0, 0, ())

        for pos in range(end + 1):
            cursor = 0
            while cursor < len(agenda[pos]):
                lhs, prod, dot, origin, children = agenda[pos][cursor]
                cursor += 1

                # Luật con đã khớp xong, cập nhật luật cha
                if dot == len(prod):
                    node = ParseNode(lhs, children)
                    for p_lhs, p_prod, p_dot, p_orig, p_ch in list(chart[origin]):
                        if p_dot < len(p_prod) and p_prod[p_dot] == ("N", lhs):
                            add(pos, p_lhs, p_prod, p_dot + 1, p_orig, p_ch + (node,))
                    continue

                kind, sym = prod[dot]
                # Mở rộng Nonterminal
                if kind == "N":
                    for child_prod in self.rules.get(sym, ()):
                        add(pos, sym, child_prod, 0, pos, ())
                    # Xử lý luật rỗng/nullable đã xong trước đó tại vị trí này
                    for d_lhs, d_prod, d_dot, d_orig, d_ch in list(chart[pos]):
                        if d_lhs == sym and d_orig == pos and d_dot == len(d_prod):
                            add(pos, lhs, prod, dot + 1, origin, children + (ParseNode(d_lhs, d_ch),))
                # So khớp xâu ký tự Terminal
                else:
                    m_end = pos + len(sym)
                    if (m_end <= end and text[pos:m_end].casefold() == sym.casefold()
                            and (m_end == end or text[m_end].isspace() or text[m_end] in "?!,.;:")):
                        add(m_end, lhs, prod, dot + 1, origin, children + (ParseNode(sym),))

        # Kiểm tra trạng thái chấp nhận (start rule S bắt đầu từ vị trí 0)
        final_pos = self._skip_spaces(text, end)
        for lhs, prod, dot, orig, children in agenda[final_pos]:
            if lhs == start and dot == len(prod) and orig == 0:
                return children[0] if children else ParseNode("S")
        return None

    def parse(self, sentence):
        """Return the first parse tree, or None if the sentence is not in the CFG."""
        normalized = unicodedata.normalize("NFC", sentence.strip())
        return self._parse_cached(normalized)

    def parse_line(self, sentence):
        node = self.parse(sentence)
        return node.to_bracketed() if node is not None else "()"

    def parse_lines(self, sentences):
        return [self.parse_line(s) for s in sentences]

    def parse_file(self, input_path=None, output_path=None):
        """Parse a UTF-8 file and write one tree (or ``()``) per input line."""
        app_dir = Path(__file__).resolve().parent
        in_p = Path(input_path) if input_path else app_dir / "input" / "sentences.txt"
        out_p = (
            Path(output_path)
            if output_path
            else app_dir / "output" / "parse-results.txt"
        )
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with in_p.open(encoding="utf-8") as src:
            results = self.parse_lines(src.read().splitlines())
        with out_p.open("w", encoding="utf-8", newline="\n") as tgt:
            tgt.write("\n".join(results) + ("\n" if results else ""))
        return results


def parse_file(input_path=None, output_path=None, grammar_path=None):
    """Convenience entry point for the assignment's file pipeline."""
    return CourseParser(grammar_path).parse_file(input_path, output_path)
