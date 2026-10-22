
from iaslab.nlp.app.course_grammar import CourseGrammar
from iaslab.nlp.app.sentence_generator import generate_to_file

generate_to_file(CourseGrammar("vi"), "output/samples.txt")

    