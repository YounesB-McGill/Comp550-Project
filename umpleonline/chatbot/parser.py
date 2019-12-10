from typing import List

import inflect
from nltk import word_tokenize, ne_chunk, pos_tag
from nltk import RegexpParser
from nltk.parse import CoreNLPParser
from nltk.tree import Tree
from nltk.util import breadth_first

from data import ALL_SENTENCES
from utils import first_letter_lowercase, first_letter_uppercase


NP_GRAMMAR = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}
        {<NNP>+}
"""
CP = RegexpParser(NP_GRAMMAR)

parser = CoreNLPParser(url='http://localhost:9000')

inflect = inflect.engine()


def get_chunks(message_text: str) -> Tree:
    """
    Return the parse given by the Stanford CoreNLP parser.
    """
    try:
        parse_list = parser.parse(message_text.split())
        for tree in parse_list:
            return tree
    except Exception as e:
        print("The following exception occurred when attempting to connect to the Stanford NLP server:\n", e)
        print("\nReturning the default NLTK parse of a sentence instead.")
        return get_chunks_fallback(message_text)


def get_chunks_fallback(message_text: str) -> Tree:
    """
    Return the default NLTK parse of a sentence. Used as a backup when the more accurate Stanford NLP server is not running.
    """
    tagged_sentence = pos_tag(word_tokenize(message_text))
    return CP.parse(tagged_sentence)


def get_NP_subtrees(tree: Tree) -> List[Tree]:
    """
    Return the lowest level, non nested NP subtrees of the input. Works with most sentences.
    """
    result = []
    all_nps = {}
    for t in tree.subtrees(lambda t: t.label() == "NP"):
        all_nps[get_tree_words(t)] = t

    for candidate in all_nps.items():
        eligible = True
        for other in all_nps.keys():
            if other == candidate[0]:
                continue
            if other in candidate[0]:
                eligible = False
                break
        if eligible:
            result.append(candidate[1])

    return result

def get_num_nonnested_NP_subtrees(tree: Tree) -> int:
    return len(get_NP_subtrees(tree))


def get_noun_from_np(np_tree: Tree) -> str:
    result = ""

    for subtree in np_tree.subtrees():
        if type(subtree[0]) == str:
            if subtree.label() in ["DT", "PRP", "PRP$", ",", "QP", "CC", "CD", "JJR"]:
                continue
            if subtree.label() == "NNS" and inflect.singular_noun(subtree[0]):
                result += first_letter_uppercase(inflect.singular_noun(subtree[0]))
            else:
                result += first_letter_uppercase(subtree[0])

    return result


def get_tree_words(tree: Tree) -> str:
    """
    Return the words of a tree node, eg:
    (NP (DT a) (JJ specific) (NN flight)) -> "a specific flight"
    """
    return " ".join(tree.leaves())


def generate_parse_trees():
    """
    Generate the parse trees found in data.py
    """
    result = []
    for s in ALL_SENTENCES:
        result.append('"""' + str(get_chunks(s)) + '""", ')
    return result
