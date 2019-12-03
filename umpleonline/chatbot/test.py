#!/usr/bin/python3

"""
Run these unit tests by running `pytest test.py` in the umpleonline/chatbot directory.
"""

import nltk

from processresponse import process_response_baseline, get_detected_keywords, get_chunks

ADD_EXAMPLE_SENTENCES = ["Add person.", "Add work in person.", "Add numeric age in person.", "Create a school."]
CONTAIN_EXAMPLE_SENTENCES = ["The house is made of rooms.", "Students contains a numeric identifier."]
HAVE_EXAMPLE_SENTENCES = ["Bulky packages are characterized by their width, length and height.",
	"Students have a numeric identifier.", "Medicines have an active ingredient."]


def test_get_detected_keywords():
    for s in ADD_EXAMPLE_SENTENCES:
        assert "ADD" in get_detected_keywords(s)
        print(get_detected_keywords(s))
    for s in CONTAIN_EXAMPLE_SENTENCES:
        assert "CONTAIN" in get_detected_keywords(s)
        print(get_detected_keywords(s))
    for s in HAVE_EXAMPLE_SENTENCES:
        assert "HAVE" in get_detected_keywords(s)
        print(get_detected_keywords(s))


def test_get_chunks():
    for s in ADD_EXAMPLE_SENTENCES:
        print(get_chunks(s))


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')


def test():
    #test_get_detected_keywords()
    test_get_chunks()


if __name__ == "__main__":
    setup_deps()
    test()
