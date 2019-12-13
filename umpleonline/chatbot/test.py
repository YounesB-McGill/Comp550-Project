#!/usr/bin/python3

"""
Run these unit tests by running `pipenv run pytest test.py -W ignore::DeprecationWarning:`
in the umpleonline/chatbot directory.

Tests marked with * are ones that pass with our system but fail with Socio. 
"""
import json
from itertools import chain
from random import shuffle
from typing import List, Tuple

import nltk
import pytest
from nltk.tree import Tree

from action import (add_class_json, add_attribute, create_association, create_inheritance, create_composition,
    return_error_to_user)
from data import (ADD_EXAMPLE_SENTENCES, CONTAIN_EXAMPLE_SENTENCES, HAVE_EXAMPLE_SENTENCES, ISA_EXAMPLE_SENTENCES,
    ALL_SENTENCES, PARSE_TREES)
from npparser import get_chunks, get_NP_subtrees, get_num_nonnested_NP_subtrees, get_noun_from_np, get_tree_words
from processresponse import (process_response_baseline, process_response_model, process_response_fallback, handle_add_kw,
    handle_contain_kw, handle_have_kw, handle_isa_kw, handle_no_kw, add_class, reset_classes_created)
from utils import get_DT_for_word, get_detected_keywords, is_attribute


TEST_DATA_PATH = "data/dataset.csv"


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('words')


def setup_function():
    reset_classes_created()


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
    for s in ISA_EXAMPLE_SENTENCES:
        assert "ISA" in get_detected_keywords(s)
        print(get_detected_keywords(s))


def test_get_DT_for_word():
    for w in ["school", "Firefighter", "PlayingCard"]:
        assert "a" == get_DT_for_word(w)
    for w in ["Application", "Elevator", "institution", "offering", "User"]:
        assert "an" == get_DT_for_word(w)


def test_is_attribute():
    attributes = ["name", "phoneNumber", "serialNumber", "competitionDate", "userId", "vehicleIdentifierNumber"]
    non_attributes = ["", "student", "course", "rgbColor", "playingCards"]

    for a in attributes:
        assert is_attribute(a)

    for na in non_attributes:
        assert not is_attribute(na)


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
    validate("BankAccount", "(NP (QP (CD one) (CC or) (JJR more)) (NN bank) (NNS accounts))")

    validate("PoliceInformationSystem", "(NP (DT The) (NN police) (NN information) (NN system))")
    validate("TeleportActionCard", "(NP (DT a) (NN teleport) (NN action) (NN card))")
    validate("LoseTurnActionCard", "(NP (DT a) (VBP lose) (NP (NN turn) (NN action) (NN card)))")


def test_get_num_nonnested_NP_subtrees():
    def validate(expected, actual):
        assert expected == get_num_nonnested_NP_subtrees(Tree.fromstring(actual))

    totree = lambda user_input: str(get_chunks(user_input))

    validate(1, "(NP (NNS tomatoes))")
    validate(1, "(NP (DT a) (JJ specific) (NN flight))")
    validate(1, "(NP (DT a) (NN teleport) (NN action) (NN card))")
    validate(1, "(NP (DT a) (VBP lose) (NP (NN turn) (NN action) (NN card)))")
    
    validate(2, """(S (NP (DT a) (VBP lose) (NP (NN turn) (NN action) (NN card)))
                   (CC and)
                   (NP (DT a) (NN teleport) (NN action) (NN card)))""")

    validate(2, """(NP (NP (DT a) (VBP lose) (NP (NN turn) (NN action) (NN card)))
                   (CC and)
                   (NP (DT a) (NN teleport) (NN action) (NN card)))""")

    validate(1, totree("Add a school."))
    validate(1, totree("Create a television."))
    validate(2, totree("Add work in person."))
    validate(2, totree("Add numeric age in person."))

    validate(2, totree("The house is made of rooms."))
    validate(2, totree("Students contains a numeric identifier."))

    validate(4, totree("Bulky packages are characterized by their width, length and height."))
    validate(2, totree("Students have a numeric identifier."))
    validate(2, totree("Medicines have an active ingredient."))

    validate(0, totree("add."))
    validate(0, totree("create."))
    validate(0, totree("make."))


def test_get_tree_words():
    def validate(expected, actual):
        assert expected == get_tree_words(Tree.fromstring(actual))

    validate("tomatoes", "(NP (NNS tomatoes))")
    validate("a specific flight", "(NP (DT a) (JJ specific) (NN flight))")
    validate("a teleport action card", "(NP (DT a) (NN teleport) (NN action) (NN card))")
    validate("a lose turn action card", "(NP (DT a) (VBP lose) (NP (NN turn) (NN action) (NN card)))")


# Pairs preceded with a double # fail when the intent has to be detected by process_response_baseline.   

ADD_PAIRS = [
    (add_class("School"), "create a school"),
    (add_class("PlayingCard"), "create a playing card"),
    (add_class("Alumnus"), "create an alumni"),

    (add_attribute("Person", "work"), "Add work in person."),
    (add_attribute("Person", "numericAge"), "Add numeric age in person."),

    # This test is disabled because Stanford NLP gives a wrong parse
    # validate(add_class("LoseTurnActionCard"), "make a lose turn action card"),
]


CONTAIN_PAIRS = [
    (create_composition("House", "Room"), "The house is made of rooms."),
    (create_composition("Car", "Wheel"), "A car contains wheels."),
    (create_composition("Car", "Tire"), "A car is composed of tires."),  # *
    (create_composition("SpecificConference", "ProgramCommittee"),
     "A specific conference consists of a program committee."),  # *

    ##(add_attribute("Student", "numericIdentifier"), "Students contains a numeric identifier."),
]


HAVE_PAIRS = [
    (add_attribute("Student", "email"), "A student has an email."),
    (add_attribute("Student", "email"), "A student is identified by their email."),
    ##(add_attribute("Student", "name"), "students are characterized by their names."),
    ##(add_attribute("Vehicle", "vehicleIdentifierNumber"),
    # "vehicles are identified by a vehicle identifier number (VIN)."),  # *
    (add_attribute("Tournament", "date"), "Each tournament has a date when it is played."),
    ##(add_attribute("Person", "phone"), "People registered are identified by their phone, which is unique."), # *
    ##(create_association("Person", "Address"), "People have addresses."),
    (create_association("Customer", "BankAccount"), "A customer has one or more bank accounts."),  # *
    ##(create_association("Student", "Major"), "Students can have one or two majors."),
    (create_association("Tileo", "PlayingCard"), "Tileo is characterized by playing cards, each with a color."),
    (return_error_to_user("I really don't understand what you meant. Please rephrase."), "Has."),
    ##(return_error_to_user("Are trying to add a class? Try saying 'Create a Car.'"), "A car has."),
]


ISA_PAIRS = [
    (create_inheritance("Student", "Person"), "A student is a person."),
    (create_inheritance("Student", "Person"), "Students are people."),
    (create_inheritance("SpecificFlight", "Flight"), "A specific flight is a flight."),
    (create_inheritance("Triangle", "Shape"), "Triangles are a type of shape."),
    (create_inheritance("Circle", "Shape"), "Circles are kinds of shapes."),
    (create_inheritance("Supervisor", "Employee"), "Some employees play roles as supervisors."),
    (create_inheritance("Lion", "GiantCat"), "A lion may be considered as a giant cat."),
    (create_inheritance("Tiger", "GiantCat"), "A tiger may also be considered a kind of giant cat."),
    (create_inheritance("TeleportActionCard", "PlayingCard"), "A teleport action card is a playing card."),

    (create_inheritance("Defender", "Player"), "Players can serve as defenders."),
    (create_inheritance("Grader", "TeachingAssistant"), "Some teaching assistants also play the role of graders."),

    (return_error_to_user("If you're trying to create an inheritance, clearly specify both classes."),
             "This item may be considered."),
]


NO_KW_PAIRS = [
    (add_class("Customer"), "Customers get hungry if not fed."),
    (add_class("BusTransportationManagementSystem"), "A bus transportation management system is used to help organize."),

    (create_association("Student", "Exam"), "Students passed exams."),
    (create_association("Check", "Bank"), "Checks get sent to the bank."),
]


def test_handle_add_kw():
    def validate(expected, actual):
        assert expected == handle_add_kw(actual)

    for pair in ADD_PAIRS:
        validate(*pair)


def test_handle_contain_kw():
    def validate(expected, actual):
        assert expected == handle_contain_kw(actual)

    for pair in CONTAIN_PAIRS:
        validate(*pair)


def test_handle_have_kw():
    def validate(expected, actual):
        assert expected == handle_have_kw(actual)

    for pair in HAVE_PAIRS:
        validate(*pair)

def test_handle_isa_kw():
    def validate(expected, actual):
        assert expected == handle_isa_kw(actual)

    for pair in ISA_PAIRS:
        validate(*pair)


def test_handle_no_kw():
    def validate(expected, actual):
        assert expected == handle_no_kw(actual)

    for pair in NO_KW_PAIRS:
        validate(*pair)


def test_process_response_baseline():
    def validate(expected, actual):
        assert expected == process_response_baseline(actual)

    for pair in chain(ADD_PAIRS, CONTAIN_PAIRS, HAVE_PAIRS, ISA_PAIRS, NO_KW_PAIRS):
        validate(*pair)


def run_all():
    """
    Run all tests.
    """
    test_get_detected_keywords()
    test_get_DT_for_word()
    test_is_attribute()
    test_get_chunks()
    test_serialized_parse_trees()
    test_get_NP_subtrees()
    test_get_noun_from_np()
    test_get_num_nonnested_NP_subtrees()
    test_get_tree_words()
    test_handle_add_kw()
    test_handle_contain_kw()
    test_handle_have_kw()
    test_handle_isa_kw()
    test_handle_no_kw()
    test_process_response_baseline()


if __name__ == "__main__":
    setup_deps()
    run_all()
