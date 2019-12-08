#!/usr/bin/python3

"""
Run these unit tests by running `pytest test.py` in the umpleonline/chatbot directory.
"""

import nltk

from data import (ADD_EXAMPLE_SENTENCES, CONTAIN_EXAMPLE_SENTENCES, HAVE_EXAMPLE_SENTENCES, ALL_SENTENCES, PARSE_TREES)
from itertools import chain
from nltk.tree import Tree
from processresponse import process_response_baseline, get_detected_keywords, get_chunks, get_NP_subtrees, get_noun_from_np


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
    for s in ALL_SENTENCES:
        c = get_chunks(s)
        assert type(c) == nltk.tree.Tree
        print(c)
        c.pretty_print()


def test_serialized_parse_trees():
    """
    Same as above test, but does not require the Stanford NLP server to run, which saves memory.
    """
    for s in PARSE_TREES:
        tree = Tree.fromstring(s)
        assert type(tree) == nltk.tree.Tree
        print(tree)
        tree.pretty_print()


def test_get_NP_subtrees():
    for s in PARSE_TREES:
        np_subtrees = get_NP_subtrees(Tree.fromstring(s))
        for t in np_subtrees:
            assert t.label() == "NP"
            print(t)


def test_get_noun_from_np():
    def validate(expected, actual):
        assert expected == get_noun_from_np(Tree.fromstring(actual))

    validate("Person", "(NP (NN person))")
    validate("Room", "(NP (NNS rooms))")

    # inflect fails to get the correct singular for curricula, radii, or indices,
    # but otherwise does well on these non-standard plurals
    validate("Wife", "(NP (NNS wives))")
    validate("Wolf", "(NP (NNS wolves))")
    validate("Tomato", "(NP (NNS tomatoes))")
    validate("Veto", "(NP (NNS vetoes))")
    validate("Foot", "(NP (NNS feet))")
    validate("Moose", "(NP (NNS moose))")
    validate("Child", "(NP (NNS children))")
    validate("Person", "(NP (NNS people))")
    validate("Alumnus", "(NP (NNS alumni))")
    validate("Thesis", "(NP (NNS theses))")
    validate("Criterion", "(NP (NNS criteria))")

    validate("School", "(NP (DT a) (NN school))")
    validate("SpecificFlight", "(NP (DT a) (JJ specific) (NN flight))")
    validate("StudentRole", "(NP (DT a) (NN student) (NN role))")
    validate("Score", "(NP (PRP$ their) (NNS scores))")

    validate("PoliceInformationSystem", "(NP (DT The) (NN police) (NN information) (NN system))")


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')


def test():
    #test_get_detected_keywords()
    #test_get_chunks()
    #test_serialized_parse_trees()
    #test_get_NP_subtrees()
    test_get_noun_from_np()


if __name__ == "__main__":
    setup_deps()
    test()
