#!/usr/bin/python3

from itertools import chain


ADD_WORDS = ["add", "create", "make"]
CONTAINS_WORDS = ["contain", "made of", "made up of", "made from", "compose", "include", "consist"]
HAVE_WORDS = ["hav", "has", "characteri", "identif", "recogni"]
ISA_WORDS = ["is a", "are", "can ", "could ", "may "]

ADD_EXAMPLE_SENTENCES = ["Add person.", "Add work in person.", "Add numeric age in person.", "Create a school."]
CONTAIN_EXAMPLE_SENTENCES = ["The house is made of rooms.", "Students contains a numeric identifier."]
HAVE_EXAMPLE_SENTENCES = ["Bulky packages are characterized by their width, length and height.",
	"Students have a numeric identifier.", "Medicines have an active ingredient."]
ISA_EXAMPLE_SENTENCES = ["A student is a person.", "Students are people.", "A specific flight is a flight.",
    "Triangles are a type of shape.", "Players can serve as defenders.", "A lion may be considered as a giant cat."]

ALL_SENTENCES = chain(ADD_EXAMPLE_SENTENCES, CONTAIN_EXAMPLE_SENTENCES, HAVE_EXAMPLE_SENTENCES, ISA_EXAMPLE_SENTENCES)

PARSE_TREES = [
    """(ROOT (S (VP (VB Add) (NP (NN person))) (. .)))""", 
    """(ROOT
    (S
        (VP (VB Add) (NP (NP (NN work)) (PP (IN in) (NP (NN person)))))
        (. .)))""", 
    """(ROOT
    (S
        (VP
        (VB Add)
        (NP (NP (JJ numeric) (NN age)) (PP (IN in) (NP (NN person)))))
        (. .)))""", 
    """(ROOT (S (VP (VB Create) (NP (DT a) (NN school))) (. .)))""", 
    """(ROOT
    (S
        (NP (DT The) (NN house))
        (VP (VBZ is) (VP (VBN made) (PP (IN of) (NP (NNS rooms)))))
        (. .)))""", 
    """(ROOT
    (S
        (NP (NNS Students))
        (VP (VBZ contains) (NP (DT a) (JJ numeric) (NN identifier)))
        (. .)))""", 
    """(ROOT
    (S
        (NP (JJ Bulky) (NNS packages))
        (VP
        (VBP are)
        (VP
            (VBN characterized)
            (PP
            (IN by)
            (NP
                (NP (PRP$ their) (NN width))
                (, ,)
                (NP (NP (NN length)) (CC and) (NP (NN height)))))))
        (. .)))""", 
    """(ROOT
    (S
        (NP (NNS Students))
        (VP (VBP have) (NP (DT a) (JJ numeric) (NN identifier)))
        (. .)))""", 
    """(ROOT
    (S
        (NP (NNP Medicines))
        (VP (VBP have) (NP (DT an) (JJ active) (NN ingredient)))
        (. .)))""", 
]
