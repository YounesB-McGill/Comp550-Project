#!/usr/bin/python3

"""
Run these unit tests by running `pipenv run pytest test.py -W ignore::DeprecationWarning:`
in the umpleonline/chatbot directory.

Tests marked with * are ones that pass with our system but fail with Socio. 
"""
from itertools import chain

import nltk
import pytest
from nltk.tree import Tree

from action import (add_class_json, add_attribute, create_association, create_inheritance, create_composition,
    return_error_to_user)
from data import (ADD_EXAMPLE_SENTENCES, CONTAIN_EXAMPLE_SENTENCES, HAVE_EXAMPLE_SENTENCES, ISA_EXAMPLE_SENTENCES,
    ALL_SENTENCES, PARSE_TREES)
from npparser import get_chunks, get_NP_subtrees, get_num_nonnested_NP_subtrees, get_noun_from_np, get_tree_words
from processresponse import (process_response_baseline, handle_add_kw, handle_contain_kw, handle_have_kw,
    add_class, reset_classes_created)
from utils import get_DT_for_word, get_detected_keywords, is_attribute


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
    assert "a" == get_DT_for_word("school")
    assert "a" == get_DT_for_word("Firefighter")
    assert "a" == get_DT_for_word("PlayingCard")

    assert "an" == get_DT_for_word("Application")
    assert "an" == get_DT_for_word("Elevator")
    assert "an" == get_DT_for_word("institution")
    assert "an" == get_DT_for_word("offering")
    assert "an" == get_DT_for_word("User")


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


def test_handle_add_kw():
    def validate(expected, actual):
        assert expected == handle_add_kw(actual)

    validate(add_class("School"), "create a school")
    validate(add_class("PlayingCard"), "create a playing card")
    validate(add_class("Alumnus"), "create an alumni")

    validate(add_attribute("Person", "work"), "Add work in person.")
    validate(add_attribute("Person", "numericAge"), "Add numeric age in person.")

    # This test is disabled because Stanford NLP gives a wrong parse
    # validate(add_class("LoseTurnActionCard"), "make a lose turn action card")


def test_handle_contain_kw():
    def validate(expected, actual):
        assert expected == handle_contain_kw(actual)

    validate(create_composition("House", "Room"), "The house is made of rooms.")
    validate(create_composition("Car", "Wheel"), "A car contains wheels.")
    validate(create_composition("Car", "Tire"), "A car is composed of tires.")  # *
    validate(create_composition("SpecificConference", "ProgramCommittee"),
             "A specific conference consists of a program committee.")  # *

    validate(add_attribute("Student", "numericIdentifier"), "Students contains a numeric identifier.")


def test_handle_have_kw():
    def validate(expected, actual):
        assert expected == handle_have_kw(actual)
    
    attributes = ["name", "phoneNumber", "serialNumber", "competitionDate", "userId", "vehicleIdentifierNumber"]
    non_attributes = ["", "student", "course", "rgbColor", "playingCards"]

    validate(add_attribute("Student", "email"), "A student has an email.")
    validate(add_attribute("Student", "email"), "A student is identified by their email.")
    validate(add_attribute("Student", "name"), "students are characterized by their names.")
    validate(add_attribute("Vehicle", "vehicleIdentifierNumber"),
             "vehicles are identified by a vehicle identifier number (VIN).")  # *
    validate(add_attribute("Tournament", "date"), "Each tournament has a date when it is played.")
    validate(add_attribute("Person", "phone"), "People registered are identified by their phone, which is unique.") # *

    validate(create_association("Person", "Address"), "People have addresses.")
    validate(create_association("Customer", "BankAccount"), "A customer has one or more bank accounts.")  # *
    validate(create_association("Student", "Major"), "Students can have one or two majors.")
    validate(create_association("Tileo", "PlayingCard"), "Tileo is characterized by playing cards, each with a color.")

    validate(return_error_to_user("I really don't understand what you meant. Please rephrase."), "Has.")
    validate(return_error_to_user("Are trying to add a class? Try saying 'Create a Car.'"), "A car has.")


def test_get_tree_words():
    def validate(expected, actual):
        assert expected == get_tree_words(Tree.fromstring(actual))

    validate("tomatoes", "(NP (NNS tomatoes))")
    validate("a specific flight", "(NP (DT a) (JJ specific) (NN flight))")
    validate("a teleport action card", "(NP (DT a) (NN teleport) (NN action) (NN card))")
    validate("a lose turn action card", "(NP (DT a) (VBP lose) (NP (NN turn) (NN action) (NN card)))")


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')


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


if __name__ == "__main__":
    setup_deps()
    run_all()
