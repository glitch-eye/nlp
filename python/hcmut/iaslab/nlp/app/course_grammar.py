"""Bilingual sentence production rules for the Course Assistant domain.

Rules describe form only. ``course_lexicon.TERMINALS`` supplies the concrete
terminal phrases. Each grammar is finite and selected explicitly by language.
"""

from iaslab.nlp.app.course_lexicon import TERMINALS
from iaslab.nlp.app.course_lexicon import CourseLexicon, DEFAULT_GRAMMAR_PATH
from iaslab.nlp.app.grammar_file import load_grammar


PRODUCTIONS = {
    "vi": {
        "S": [["Q_COURSE_INFO"], ["Q_TOPIC"], ["Q_SCHEDULE"],
              ["Q_ASSIGNMENT"], ["Q_DEADLINE"], ["Q_RESOURCE"], ["Q_RULE"]],
        "Q_COURSE_INFO": [
            ["COURSE_NP", "COURSE_INFO_TAIL"],
        ],
        "COURSE_INFO_TAIL": [["có bao nhiêu tín chỉ?"], ["gồm những phần nào?"],
                              ["có những nội dung gì?"]],
        "Q_TOPIC": [
            ["CHAPTER_NP", "TOPIC_TAIL"],
            ["CHAPTER_WH", "V_TOPIC", "TOPIC_NP", "?"],
            ["TOPIC_NP", "TOPIC_WHERE"],
        ],
        "TOPIC_TAIL": [["nói về gì?"], ["có những chủ đề nào?"],
                       ["đề cập đến nội dung gì?"]],
        "CHAPTER_WH": [["Chương nào"]],
        "TOPIC_WHERE": [["được học ở đâu?"]],
        "Q_SCHEDULE": [
            ["WEEK_NP", "SCHEDULE_TAIL"],
            ["CHAPTER_NP", "SCHEDULE_CHAPTER_TAIL"],
            ["SCHEDULE_WHEN"],
        ],
        "SCHEDULE_TAIL": [["học nội dung gì?"], ["có lịch học khi nào?"]],
        "SCHEDULE_CHAPTER_TAIL": [["được học vào tuần nào?"],
                                   ["học vào tuần nào?"]],
        "SCHEDULE_WHEN": [["Lịch học diễn ra khi nào?"]],
        "Q_ASSIGNMENT": [["ASSIGNMENT_NP", "ASSIGNMENT_TAIL"]],
        "ASSIGNMENT_TAIL": [["yêu cầu làm gì?"], ["gồm những phần nào?"],
                             ["có phần hỏi đáp không?"]],
        "Q_DEADLINE": [["DEADLINE_NP", "DEADLINE_TAIL"],
                       ["ASSIGNMENT_NP", "DEADLINE_ASSIGNMENT_TAIL"]],
        "DEADLINE_TAIL": [["là khi nào?"], ["được quy định như thế nào?"]],
        "DEADLINE_ASSIGNMENT_TAIL": [["nộp khi nào?"]],
        "Q_RESOURCE": [["RESOURCE_NP", "RESOURCE_TAIL"],
                       ["RESOURCE_TOPIC_PREFIX", "TOPIC_NP", "?"]],
        "RESOURCE_TAIL": [["gồm những gì?"], ["nào được sử dụng?"]],
        "RESOURCE_TOPIC_PREFIX": [["Có tài liệu nào về"]],
        "Q_RULE": [["RULE_NP", "RULE_TAIL"],
                   ["RULE_PREFIX", "RULE_OBJECT", "không?"]],
        "RULE_TAIL": [["là gì?"], ["quy định như thế nào?"]],
        "RULE_PREFIX": [["Có được sử dụng"]],
    },
    "en": {
        "S": [["Q_COURSE_INFO"], ["Q_TOPIC"], ["Q_SCHEDULE"],
              ["Q_ASSIGNMENT"], ["Q_DEADLINE"], ["Q_RESOURCE"], ["Q_RULE"]],
        "Q_COURSE_INFO": [["How many credits does", "COURSE_NP", "have?"],
                          ["What topics does", "COURSE_NP", "cover?"]],
        "Q_TOPIC": [
            ["CHAPTER_NP", "TOPIC_TAIL"],
            ["CHAPTER_WH", "V_TOPIC", "TOPIC_NP", "?"],
            ["TOPIC_NP", "TOPIC_WHERE"],
        ],
        "TOPIC_TAIL": [["cover?"], ["discuss?"], ["include?"]],
        "CHAPTER_WH": [["Which chapter"]],
        "TOPIC_WHERE": [["is taught in which chapter?"]],
        "Q_SCHEDULE": [["WEEK_NP", "SCHEDULE_TAIL"],
                       ["CHAPTER_NP", "SCHEDULE_CHAPTER_TAIL"],
                       ["SCHEDULE_WHEN"]],
        "SCHEDULE_TAIL": [["covers which topics?"], ["is scheduled when?"]],
        "SCHEDULE_CHAPTER_TAIL": [["is taught in which week?"]],
        "SCHEDULE_WHEN": [["When is the course scheduled?"]],
        "Q_ASSIGNMENT": [["ASSIGNMENT_NP", "ASSIGNMENT_TAIL"]],
        "ASSIGNMENT_TAIL": [["requires what?"], ["include which parts?"],
                             ["include a question answering component?"]],
        "Q_DEADLINE": [["DEADLINE_NP", "DEADLINE_TAIL"]],
        "DEADLINE_TAIL": [["is when?"], ["is specified where?"]],
        "Q_RESOURCE": [["RESOURCE_PREFIX", "TOPIC_NP", "?"],
                       ["RESOURCE_NP", "RESOURCE_TAIL"]],
        "RESOURCE_PREFIX": [["Which resources cover"]],
        "RESOURCE_TAIL": [["are available?"], ["are required?"]],
        "Q_RULE": [["RULE_NP", "RULE_TAIL"],
                   ["RULE_PREFIX", "RULE_NP", "allowed?"]],
        "RULE_TAIL": [["say what?"], ["apply to the course?"]],
        "RULE_PREFIX": [["Is"]],
    },
}


class CourseGrammar(object):
    """Adapts one language's grammar and terminals to SentenceGenerator."""

    def __init__(self, language, grammar_path=None):
        if language not in PRODUCTIONS:
            raise ValueError("language must be 'vi' or 'en'")
        self.language = language
        self.grammar_path = grammar_path or DEFAULT_GRAMMAR_PATH
        self.file_productions = (
            load_grammar(self.grammar_path) if language == "vi" else None
        )
        self.lexicon = CourseLexicon(self.grammar_path) if language == "vi" else None

    @property
    def start_symbol(self):
        return "S"

    def is_nonterminal(self, symbol):
        if self.file_productions is not None:
            return symbol in self.file_productions
        return (symbol in PRODUCTIONS[self.language]
                or symbol in TERMINALS[self.language])

    def productions_for(self, nonterminal):
        if self.file_productions is not None:
            return self.file_productions.get(nonterminal, [])
        if nonterminal in PRODUCTIONS[self.language]:
            return PRODUCTIONS[self.language][nonterminal]
        if nonterminal in TERMINALS[self.language]:
            return [[value] for value in TERMINALS[self.language][nonterminal]]
        return []
