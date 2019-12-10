import string
from typing import Dict, List

from nltk.corpus import wordnet

from data import ADD_WORDS, CONTAINS_WORDS, HAVE_WORDS, ISA_WORDS


def first_letter_uppercase(user_input: str) -> str:
    return user_input[0].upper() + user_input[1:]


def first_letter_lowercase(user_input: str) -> str:
    return user_input[0].lower() + user_input[1:]


def strip_punctuation(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation))


def contains_one_of(user_input: str, targets: List[str]) -> bool:
    """
    Return True if the input string contains any one of the targets.  
    """
    return any(w in user_input for w in targets)


def get_DT_for_word(word) -> str:
    if len(word) == 0:
        return ""
    word = word.lower()
    if word[0] in ["a", "e", "i", "o", "u"]:
        return "an"
    else:
        return "a"


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
    for w in ISA_WORDS:
        if w == " serve" and w in user_input and " as " in user_input:
            result["ISA"] = "serve"
        elif w == " play" and w in user_input and " role" in user_input:
            result["ISA"] = "play"
        else:
            if w in user_input:
                result["ISA"] = w


    return result


def is_attribute(noun: str) -> bool:
    """
    Return True if the noun is a common attribute.
    """
    noun = noun.lower()
    common_attribute_parts = ["name", "number", "identifier", "date", "time", "string"]
    for p in common_attribute_parts:
        if p in noun:
            return True
    return noun.lower() in ["email", "id", "userid", "phone", "color"]


def get_synonyms(word: str) -> set:
    result = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            result.add(lemma.name())
    return result


def get_hypernym(word: str) -> set:
    try:
        return wordnet.synsets(word)[0].hypernyms()[0].lemmas()[0].name()
    except:
        return None
