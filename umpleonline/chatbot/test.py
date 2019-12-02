#!/usr/bin/python3

"""
Run these unit tests by running `pytest test.py` in the umpleonline/chatbot directory.
"""

from processresponse import process_response_baseline, get_detected_keywords

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


def test():
    test_get_detected_keywords()


if __name__ == "__main__":
    test()
