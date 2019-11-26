#!/usr/bin/python3
import json
import re


def process_response_baseline(user_input):
    """
    Function used to reply with a baseline response based on the Socio model.
    This function assumes valid input.
    """

    # this needs to move to wherever the add rule is defined
    ADD_CLASS_WORDS = ["add", "create", "make"]
    
    print("Processing request in debug mode")
    message_text = user_input.lower()

    # need to replace this with NLTK chunking
    words = message_text.split(' ')

    if any(w in message_text for w in ADD_CLASS_WORDS):
        for i in range(len(words) - 2):
            if words[i] in ADD_CLASS_WORDS:
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


def add_class(class_name):
    return json.dumps({
        "intents": [{"intent": "create_class"}],
        "entities": [{"value": class_name}],
        "output": {"text": [f"I created a class called {class_name}."]}
    })


def add_attribute(class_name, attribute_name):
    return json.dumps({
        "intents": [{"intent": "add_attribute"}],
        "entities": [{"value": class_name}, {"value": attribute_name}],
        "output": [{"text": f"{class_name} now has the attribute {attribute_name}."}]
    })


def create_composition(whole_class_name, part_class_name):
    return json.dumps({
        "intents": [{"intent": "create_composition"}],
        "entities": [{"value": whole_class_name}, {"value": part_class_name}],
        "output": {"text": [f"{whole_class_name} is now composed of {part_class_name}."]},
        "context": {"varContainer": whole_class_name, "varPart": part_class_name}
    })


def create_association(class_name1, class_name2):
    return json.dumps({
        "intents": [{"intent": "create_association"}],
        "entities": [{"value": class_name1}, {"value": class_name2}],
        "output": [{"text": f"A {class_name1} has many {class_name2}s."}],
    })


def create_inheritance(child, parent):
    return json.dumps({
        "intents": [{"intent": "create_inheritance"}],
        "entities": [{"value": child}, {"value": parent}],
        "output": {"text": [f"{child} is a subclass of {parent}."]}
    })

def first_letter_uppercase(user_input):
    return user_input[0].upper() + user_input[1:]


def strip_punctuation(s):
    return re.sub(r"/\s+/g", " ", re.sub(r"/[^\w\s]|_/g", "", s))

