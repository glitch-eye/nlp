"""Generate sentences from the grammar module (Phần I).

The grammar module is injected through the small adapter interface below, so
this module does not need to know how Phần I stores or parses its productions.
"""

from collections import deque
from pathlib import Path
import re


class GrammarAdapter(object):
    """Interface expected from an adapter around the completed grammar module.

    Implement ``start_symbol``, ``is_nonterminal(symbol)`` and
    ``productions_for(nonterminal)``. Each production is a list of symbols.
    """

    @property
    def start_symbol(self):
        raise NotImplementedError

    def is_nonterminal(self, symbol):
        raise NotImplementedError

    def productions_for(self, nonterminal):
        raise NotImplementedError


class SentenceGenerator(object):
    MAX_SENTENCES = 10000
    DEFAULT_MAX_EXPANSIONS = 30

    def __init__(self, grammar):
        self.grammar = grammar

    def generate(self, max_sentences=MAX_SENTENCES,
                 max_expansions=DEFAULT_MAX_EXPANSIONS):
        """Enumerate derivations breadth-first and return unique sentences.

        ``max_expansions`` bounds recursion/cycles. Terminals that contain
        multiple words should be represented as one symbol by the adapter.
        """
        if not 1 <= max_sentences <= self.MAX_SENTENCES:
            raise ValueError("max_sentences must be between 1 and 10000")
        if max_expansions < 1:
            raise ValueError("max_expansions must be positive")

        pending = deque([([self.grammar.start_symbol], 0)])
        sentences = []
        seen = set()

        while pending and len(sentences) < max_sentences:
            symbols, expansion_count = pending.popleft()
            index = next(
                (i for i, symbol in enumerate(symbols)
                 if self.grammar.is_nonterminal(symbol)),
                None,
            )

            if index is None:
                sentence = " ".join(
                    symbol.strip() for symbol in symbols
                    if symbol is not None and symbol.strip()
                )
                sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence)
                if sentence and sentence not in seen:     
                    seen.add(sentence)
                    sentences.append(sentence)
                continue
            
            if expansion_count >= max_expansions:
                continue

            nonterminal = symbols[index]
            for production in self.grammar.productions_for(nonterminal) or []:
                expanded = (
                    symbols[:index] + list(production) + symbols[index + 1:]
                )
                pending.append((expanded, expansion_count + 1))

        return sentences

    @staticmethod
    def write(sentences, output_path="output/samples.txt"):
        """Write one sentence per line, encoded as UTF-8."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for sentence in sentences[:SentenceGenerator.MAX_SENTENCES]:
                stream.write(sentence + "\n")


def generate_to_file(grammar, output_path="output/samples.txt",
                     max_sentences=SentenceGenerator.MAX_SENTENCES,
                     max_expansions=SentenceGenerator.DEFAULT_MAX_EXPANSIONS):
    """Convenience entry point for the assignment pipeline."""
    generator = SentenceGenerator(grammar)
    sentences = generator.generate(max_sentences, max_expansions)
    generator.write(sentences, output_path)
    return sentences
