"""Terminal lexicon for Course Assistant sentence generation.

Grammar productions live in ``course_grammar.py``. This module contains only
terminal phrases grouped by semantic/POS category and output language.
"""

from pathlib import Path

from iaslab.nlp.app.grammar_file import load_grammar


DEFAULT_GRAMMAR_PATH = Path(__file__).with_name("output") / "grammar.txt"


class CourseLexicon(object):
    """Lexical entries read from the terminal-only rules in a CFG file."""

    def __init__(self, grammar_path=DEFAULT_GRAMMAR_PATH):
        rules = load_grammar(grammar_path)
        self.entries = {
            nonterminal: [production[0] for production in alternatives
                          if len(production) == 1
                          and production[0] not in rules]
            for nonterminal, alternatives in rules.items()
        }
        self.entries = {key: values for key, values in self.entries.items() if values}

    def entries_for(self, category):
        return self.entries.get(category, [])

TERMINALS = {
    "vi": {
        "COURSE_NP": ["môn NLP", "môn học CO3085", "Xử lý Ngôn ngữ Tự nhiên"],
        "CHAPTER_NP": ["chương 1", "chương 2", "chương 3", "chương 4",
                       "chương 5", "chương 6", "chương 7", "chương 8",
                       "chương 9", "chương 10", "chương 11", "chương 12"],
        "TOPIC_NP": ["CFG", "PCFG", "văn phạm phụ thuộc", "nét và văn phạm gia tố",
                     "gán nhãn từ loại", "dạng luận lý", "vai trò chủ đề",
                     "giới hạn lựa chọn", "mạng ngữ nghĩa", "văn phạm ngữ nghĩa",
                     "biểu diễn tri thức", "hỏi đáp", "tham chiếu", "đồng tham chiếu",
                     "cụm danh từ", "cụm động từ", "cụm tính từ", "cụm trạng từ"],
        "ASSIGNMENT_NP": ["bài tập lớn", "Phần I", "Phần II", "Phần III", "Phần IV"],
        "RESOURCE_NP": ["tài liệu tham khảo", "slide", "giáo trình", "tài liệu Python NLP"],
        "RULE_NP": ["quy định sử dụng AI", "quy định về LLM/RAG", "quy định của môn học"],
        "RULE_OBJECT": ["AI", "LLM/RAG"],
        "WEEK_NP": ["tuần 1", "tuần 2", "tuần 3", "tuần 4", "tuần 5",
                    "tuần 6", "tuần 7", "tuần 8", "tuần 9", "tuần 10",
                    "tuần 11", "tuần 12", "tuần 13", "tuần 14", "tuần 15"],
        "DEADLINE_NP": ["hạn nộp bài tập lớn", "deadline", "thời hạn nộp bài"],
        "EXAM_NP": ["midterm", "final", "bài thi giữa kỳ", "bài thi cuối kỳ"],
        "V_TOPIC": ["nói về", "đề cập đến"],
        "V_SCHEDULE": ["học", "được học"],
        "V_ASSIGNMENT": ["yêu cầu", "gồm"],
        "PREPOSITION": ["về", "trong", "ở", "vào", "của"],
        "ADVERB_WH": ["khi nào", "ở đâu", "như thế nào"],
        "QUANTIFIER": ["những", "các", "mấy", "bao nhiêu"],
    },
    "en": {
        "COURSE_NP": ["NLP", "course CO3085", "Natural Language Processing"],
        "CHAPTER_NP": ["chapter 1", "chapter 2", "chapter 3", "chapter 4",
                       "chapter 5", "chapter 6", "chapter 7", "chapter 8",
                       "chapter 9", "chapter 10", "chapter 11", "chapter 12"],
        "TOPIC_NP": ["CFG", "PCFG", "dependency grammar", "features and augmented grammar",
                     "part-of-speech tagging", "logical form", "thematic roles",
                     "selectional restriction", "semantic networks", "semantic grammar",
                     "knowledge representation", "question answering", "reference",
                     "anaphora", "noun phrases", "verb phrases", "adjective phrases",
                     "adverbial phrases"],
        "ASSIGNMENT_NP": ["the assignment", "Part I", "Part II", "Part III", "Part IV"],
        "RESOURCE_NP": ["the reference materials", "the slides", "the textbook", "Python NLP materials"],
        "RULE_NP": ["the AI usage policy", "the course rules"],
        "RULE_OBJECT": ["AI", "LLM/RAG"],
        "WEEK_NP": ["week 1", "week 2", "week 3", "week 4", "week 5",
                    "week 6", "week 7", "week 8", "week 9", "week 10",
                    "week 11", "week 12", "week 13", "week 14", "week 15"],
        "DEADLINE_NP": ["the assignment deadline", "the submission deadline"],
        "EXAM_NP": ["the midterm", "the final exam"],
        "V_TOPIC": ["covers", "discusses", "introduces"],
        "V_SCHEDULE": ["is taught", "covers"],
        "V_ASSIGNMENT": ["requires", "includes"],
        "PREPOSITION": ["about", "in", "during", "for"],
        "ADVERB_WH": ["when", "where", "how"],
        "QUANTIFIER": ["which", "what", "how many"],
    },
}

# POS labels for documentation/inspection. Grammar rules refer to phrase
# categories above (NP/VP-like units), since multiword expressions are terminals.
POS_BY_CATEGORY = {
    "COURSE_NP": "noun phrase",
    "CHAPTER_NP": "noun phrase",
    "TOPIC_NP": "noun phrase",
    "ASSIGNMENT_NP": "noun phrase",
    "RESOURCE_NP": "noun phrase",
    "RULE_NP": "noun phrase",
    "RULE_OBJECT": "noun phrase",
    "WEEK_NP": "noun phrase",
    "DEADLINE_NP": "noun phrase",
    "EXAM_NP": "noun phrase",
    "V_TOPIC": "verb phrase",
    "V_SCHEDULE": "verb phrase",
    "V_ASSIGNMENT": "verb phrase",
    "PREPOSITION": "preposition",
    "ADVERB_WH": "adverbial phrase",
    "QUANTIFIER": "quantifier",
}
