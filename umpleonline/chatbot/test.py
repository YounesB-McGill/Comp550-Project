#!/usr/bin/python3

"""
Run these unit tests by running `pytest test.py` in the umpleonline/chatbot directory.
"""

import nltk

from data import (ADD_EXAMPLE_SENTENCES, CONTAIN_EXAMPLE_SENTENCES, HAVE_EXAMPLE_SENTENCES, ALL_SENTENCES, PARSE_TREES)
from itertools import chain
from nltk.tree import Tree
from processresponse import process_response_baseline, get_detected_keywords, get_chunks, get_NP_subtrees


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
    get_NP_subtrees(Tree.fromstring(PARSE_TREES[0]))


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')


def test():
    #test_get_detected_keywords()
    #test_get_chunks()
    #test_serialized_parse_trees()
    test_get_NP_subtrees()



if __name__ == "__main__":
    setup_deps()
    test()
