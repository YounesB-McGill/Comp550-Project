#!/usr/bin/python3
import inflect
import json
import re

from nltk import word_tokenize, ne_chunk, pos_tag
from nltk import RegexpParser
from nltk.corpus import wordnet
from nltk.parse import CoreNLPParser
from nltk.tree import Tree
from nltk.util import breadth_first
from typing import Dict, List


# this needs to move to wherever the add rule is defined
ADD_WORDS = ["add", "create", "make"]
CONTAINS_WORDS = ["contain", "made of", "made up of", "made from", "compose", "include", "consist"]
HAVE_WORDS = ["hav", "has", "characteri", "identif", "recogni"]
NP_GRAMMAR = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}
        {<NNP>+}
"""
CP = RegexpParser(NP_GRAMMAR)

parser = CoreNLPParser(url='http://localhost:9000')
inflect = inflect.engine()

classes_created = []  # Must keep track of this to avoid errors


def process_response_baseline(user_input: str) -> str:
    """
    Function used to reply with a baseline response based on the Socio model.
    This function assumes valid input.
    """
    message_text = strip_punctuation(user_input.lower())
    detected_keywords = get_detected_keywords(message_text)
    nk = len(detected_keywords)

    if nk == 0:
        return process_response_fallback(message_text)
    elif nk == 1:
        kw = list(detected_keywords.keys())[0]
        if kw == "ADD":
            return handle_add_kw(message_text)
        elif kw == "CONTAIN":
            return handle_contain_kw(message_text)
        elif kw == "HAVE":
            return handle_have_kw(message_text)
    else:
        # TODO Handle multiple keywords, eg "Students *contain*s a numeric *identif*ier"
        return process_response_fallback(message_text)


def handle_add_kw(message_text: str) -> str:
    chunks = get_chunks(message_text)
    nps = get_NP_subtrees(chunks)
    n_st = get_num_nonnested_NP_subtrees(chunks)
    if n_st == 0:
        kw = get_detected_keywords(message_text).get("ADD", "add")
        return return_error_to_user(f"Please specify what you want to {kw}.")
    elif n_st == 1:
        class_name = get_noun_from_np(nps[0])
        return add_class(class_name)
    elif n_st == 2:
        for t in nps:
            t.pretty_print()
        class_name = get_noun_from_np(nps[1])
        attribute_name = first_letter_lowercase(get_noun_from_np(nps[0]))
        return add_attribute(class_name, attribute_name)
    else:
        return process_response_fallback(message_text)


def handle_contain_kw(message_text: str) -> str:
    chunks = get_chunks(message_text)
    nps = get_NP_subtrees(chunks)
    n_st = get_num_nonnested_NP_subtrees(chunks)
    if n_st < 2:
        return return_error_to_user(
            "I don't get what you meant. If you want to make a composition, specify the two classes.")
    elif n_st == 2:
        whole = get_noun_from_np(nps[0])
        part = get_noun_from_np(nps[1])

        if whole not in classes_created:
            classes_created.append(whole)
        if part not in classes_created:
            classes_created.append(part)

        return create_composition(whole, part)
    else:
        return process_response_fallback(message_text)


def handle_have_kw(message_text: str) -> str:
    return process_response_fallback(message_text)


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


def get_num_nonnested_NP_subtrees_(tree: Tree) -> int:
    result = 0

    for t in tree.subtrees(lambda t: t.label() == "NP"):
        result += 1

        n_NP_subtrees = 0
        for j, st in enumerate(t.subtrees(lambda t: t.label() == "NP")):
            if j == 0:
                continue

            n_NP_subsubtrees = 0
            for k, sst in enumerate(st.subtrees(lambda t: t.label() == "NP")):
                if k == 0:
                    continue
                n_NP_subsubtrees += 1
            
            if n_NP_subsubtrees in [1, 2]:
                result -= 1

            n_NP_subtrees += 1

        if n_NP_subtrees in [1, 2]:
            result -= 1

    return result


def get_noun_from_np(np_tree: Tree) -> str:
    result = ""

    for subtree in np_tree.subtrees():
        if type(subtree[0]) == str:
            if subtree.label() in ["DT", "PRP", "PRP$", ","]:
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


def get_synonyms(word: str) -> set:
    res = set()
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            res.add(lm.name())
    return res


def process_response_fallback(user_input: str) -> str:
    """
    Fallback method from Younes' undergrad project, to be used for the cases not handled by Socio's logic.
    """
    print("Processing request in fallback mode")
    message_text = user_input.lower()
    words = message_text.split(' ')

    # This logic is not always correct, eg "Add attribute in class."
    if contains_one_of(message_text, ADD_WORDS):
        for i in range(len(words) - 2):
            if words[i] in ADD_WORDS:
                # strip punctuation
                class_name = first_letter_uppercase(strip_punctuation(words[i + 2]))
                return add_class(class_name)

    if "has a" in message_text:
        for i in range(len(words) - 2):
            if words[i] == 'has':
                class_name = first_letter_uppercase(words[i - 1])
                attribute_name = strip_punctuation(words[i + 2])
                return add_attribute(class_name, attribute_name)

    if "is composed of" in message_text:
        for i in range(len(words) - 2):
            if words[i] == "is":
                whole_class_name = first_letter_uppercase(words[i - 1])
                part_class_name = first_letter_uppercase(strip_punctuation(words[i + 3]))
                # assume the plural when part_class_name ends with s
                if part_class_name[-1] == "s":
                    part_class_name = part_class_name[:-1]
                return create_composition(whole_class_name, part_class_name)

    # not very useful, but good for testing
    if "is associated with" in message_text:
        for i in range(len(words) - 3):
            if words[i] == "is":
                class_name1 = first_letter_uppercase(words[i - 1])
                if words[i + 3] in ["a", "an"]:
                    class_name2 = words[i + 4]
                else:
                    class_name2 = words[i + 3]
                class_name2 = first_letter_uppercase(strip_punctuation(class_name2))
                return create_association(class_name1, class_name2)

    if "is a" in message_text:
        for i in range(len(words) - 2):
            if words[i] == "is":
                child = first_letter_uppercase(words[i - 1])
                parent = first_letter_uppercase(strip_punctuation(words[i + 2]))
                return create_inheritance(child, parent)

    # Debug WordNet synonyms
    if "synonym of" in message_text:
        for i in range(len(words)):
            pass #if words[]

    return "Error"



def get_detected_keywords(user_input: str) -> Dict[str, str]:
    """
    Returns detected keywords used by Socio's rules.
    """
    user_input = user_input.lower()
    result = {}

    for w in ADD_WORDS:
        if w in user_input:
            result["ADD"] = w
    for w in CONTAINS_WORDS:
        if w in user_input:
            result["CONTAIN"] = w
    for w in HAVE_WORDS:
        if w in user_input:
            result["HAVE"] = w

    return result


def add_class(class_name: str) -> str:
    global classes_created
    if class_name in classes_created:
        return return_error_to_user(f"{class_name} is already created, so let's not make it again.")
    
    return add_class_json(class_name)


def add_class_json(class_name: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_class"}],
        "entities": [{"value": class_name}],
        "output": {"text": [f"I created a class called {class_name}."]}
    })


def add_attribute(class_name: str, attribute_name: str) -> str:
    return json.dumps({
        "intents": [{"intent": "add_attribute"}],
        "entities": [{"value": class_name}, {"value": attribute_name}],
        "output": [{"text": f"{class_name} now has the attribute {attribute_name}."}]
    })


def create_composition(whole_class_name: str, part_class_name: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_composition"}],
        "entities": [{"value": whole_class_name}, {"value": part_class_name}],
        "output": {"text": [f"{whole_class_name} is now composed of {part_class_name}."]},
        "context": {"varContainer": whole_class_name, "varPart": part_class_name}
    })


def create_association(class_name1: str, class_name2: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_association"}],
        "entities": [{"value": class_name1}, {"value": class_name2}],
        "output": [{"text": f"A {class_name1} has many {class_name2}s."}],
    })


def create_inheritance(child: str, parent: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_inheritance"}],
        "entities": [{"value": child}, {"value": parent}],
        "output": {"text": [f"{child} is a subclass of {parent}."]}
    })


def return_error_to_user(error_msg: str) -> str:
    return json.dumps({
        "intents": [{"intent": "return_error_to_user"}],
        "output": {"text": error_msg}
    })


def reset_classes_created():
    global classes_created
    classes_created = []


def first_letter_uppercase(user_input: str) -> str:
    return user_input[0].upper() + user_input[1:]


def first_letter_lowercase(user_input: str) -> str:
    return user_input[0].lower() + user_input[1:]


def strip_punctuation(s: str) -> str:
    return re.sub(r"/\s+/g", " ", re.sub(r"/[^\w\s]|_/g", "", s))


def contains_one_of(user_input: str, targets: str) -> bool:
    """
    Return True if the input string contains any one of the targets.  
    """
    return any(w in user_input for w in targets)

