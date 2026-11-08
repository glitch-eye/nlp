"""Adapter exposing the Course Assistant CFG to the sentence generator."""

from pathlib import Path

from iaslab.nlp.app.grammar_file import load_grammar


DEFAULT_GRAMMAR_PATH = Path(__file__).with_name("output") / "grammar.txt"


class CourseGrammar(object):
    """Read the shared grammar file and implement the generator interface."""

    def __init__(self, language="vi", grammar_path=None):
        if language != "vi":
            raise ValueError("Only the Vietnamese Course Assistant grammar is configured")
        self.language = language
        self.grammar_path = Path(grammar_path or DEFAULT_GRAMMAR_PATH)
        self.productions = load_grammar(self.grammar_path)

    @property
    def start_symbol(self):
        return "S"

    def is_nonterminal(self, symbol):
        return symbol in self.productions

    def productions_for(self, nonterminal):
        return self.productions.get(nonterminal, [])
