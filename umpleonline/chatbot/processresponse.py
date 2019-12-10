#!/usr/bin/python3
import re
from typing import List

from action import (add_class_json, add_attribute, create_association, create_inheritance, create_composition,
    return_error_to_user)
from data import ADD_WORDS, CONTAINS_WORDS, HAVE_WORDS, ISA_WORDS
from npparser import get_chunks, get_NP_subtrees, get_num_nonnested_NP_subtrees, get_noun_from_np
from utils import (first_letter_lowercase, first_letter_uppercase, contains_one_of, get_DT_for_word, is_attribute,
    get_detected_keywords, strip_punctuation)


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
        return handle_no_kw(message_text)
    elif nk == 1:
        kw = list(detected_keywords.keys())[0]
        if kw == "ADD":
            return handle_add_kw(message_text)
        elif kw == "CONTAIN":
            return handle_contain_kw(message_text)
        elif kw == "HAVE":
            return handle_have_kw(message_text)
        elif kw == "ISA":
            return handle_isa_kw(message_text)
    elif nk == 2:
        if "CONTAIN" in detected_keywords.keys() and "ISA" in detected_keywords.keys(): # "can consist of"
            return handle_contain_kw(message_text)
        else:
            return process_response_fallback(message_text)
    else:
        # TODO Handle more complex multiple keyword scenarios
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
        first_noun = get_noun_from_np(nps[0])
        second_noun = get_noun_from_np(nps[1])

        if first_noun not in classes_created:
            classes_created.append(first_noun)

        if is_attribute(get_noun_from_np(nps[1])):
            return add_attribute(first_noun, first_letter_lowercase(second_noun))
        else:
            whole = first_noun
            part = second_noun

            if part not in classes_created:
                classes_created.append(part)

            return create_composition(whole, part)
    else:
        return process_response_fallback(message_text)


def handle_have_kw(message_text: str) -> str:
    chunks = get_chunks(message_text)
    nps = get_NP_subtrees(chunks)
    n_st = get_num_nonnested_NP_subtrees(chunks)
    if n_st == 0:
        return return_error_to_user("I really don't understand what you meant. Please rephrase.")
    elif n_st == 1:
        class_name = get_noun_from_np(nps[0])
        if class_name in classes_created:
            return return_error_to_user(f"What do want to specify about {class_name}?")
        else:
            dt = get_DT_for_word(class_name)
            return return_error_to_user(f"Are trying to add a class? Try saying 'Create {dt} {class_name}.'")
    else:
        # TODO In the future, also allow multiple attributes ("Student has a name and email").
        # This requires updating the website.
        class_name = get_noun_from_np(nps[0])
        second_noun = get_noun_from_np(nps[1])

        if class_name in classes_created:
            classes_created.append(class_name)

        if is_attribute(second_noun):
            return add_attribute(class_name, first_letter_lowercase(second_noun))
        else:
            if second_noun not in classes_created:
                classes_created.append(second_noun)
            return create_association(class_name, second_noun)

    return process_response_fallback(message_text)


def handle_isa_kw(message_text: str) -> str:
    chunks = get_chunks(message_text)
    nps = get_NP_subtrees(chunks)
    n_st = get_num_nonnested_NP_subtrees(chunks)
    if n_st < 2:
        return return_error_to_user("If you're trying to create an inheritance, clearly specify both classes.")
    else:
        if ((" serve" in message_text and " as " in message_text) or
            (" play" in message_text and " role" in message_text)):
            child = get_noun_from_np(nps[1])
            parent = get_noun_from_np(nps[0])
        else:
            child = get_noun_from_np(nps[0])
            parent = get_noun_from_np(nps[1])

        if child not in classes_created:
            classes_created.append(child)

        if parent not in classes_created:
            classes_created.append(parent)

        return create_inheritance(child, parent)

    return process_response_fallback(message_text)


def handle_no_kw(message_text: str) -> str:
    # This will add an association if possible, otherwise it will create a class
    return process_response_fallback(message_text)   


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

    return return_error_to_user("Sorry, I could not process your request :(")


# This function is kept here since it modifies the global state
def add_class(class_name: str) -> str:
    global classes_created
    if class_name in classes_created:
        return return_error_to_user(f"{class_name} is already created, so let's not make it again.")
    
    return add_class_json(class_name)


def reset_classes_created():
    global classes_created
    classes_created = []
