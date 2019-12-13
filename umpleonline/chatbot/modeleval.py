#!/usr/bin/python3
import json
from random import shuffle
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from model import predict
from processresponse import process_response_baseline, process_response_model, process_response_fallback, get_intent


"""
Run this using `pipenv run python3 modeleval.py`
in the umpleonline/chatbot directory. 
"""


TEST_DATA_PATH = "data/testData.csv"  # Contains 1000 labeled sentences
COMPLEX_TEST_DATA_PATH = "data/complexTestData.csv" # Contains 36 complex sentences


def compute_accuracy_and_f1_fallback_baseline():
    test = pd.read_csv(TEST_DATA_PATH, encoding="latin1", names=["Sentence", "Intent"])
    complexTest = pd.read_csv(COMPLEX_TEST_DATA_PATH, encoding="latin1", names=["Sentence", "Intent"])

    for f in [process_response_fallback, process_response_baseline]:
        yPred = []
        yTrue = []
        for i, j in test.iterrows():
            yPred.append(json.loads(f(j['Sentence']))["intents"][0]["intent"])
            yTrue.append(j['Intent'])

        print(f"1000 test {f.__name__} accuracy = {100*accuracy_score(yTrue, yPred)}%.\n")
        print(f"1000 test {f.__name__} F1 = {100*f1_score(yTrue, yPred, average='macro')}%.\n")

    for f in [process_response_fallback, process_response_baseline]:
        yPred = []
        yTrue = []
        for i, j in complexTest.iterrows():
            yPred.append(json.loads(f(j['Sentence']))["intents"][0]["intent"])
            yTrue.append(j['Intent'])

        print(f"Complex test {f.__name__} accuracy = {100*accuracy_score(yTrue, yPred)}%.\n")
        print(f"Complex test {f.__name__} F1 = {100*f1_score(yTrue, yPred, average='macro')}%.\n")


def compute_accuracy_and_f1_model():
    test = pd.read_csv(TEST_DATA_PATH, encoding="latin1", names=["Sentence", "Intent"])
    complexTest = pd.read_csv(COMPLEX_TEST_DATA_PATH, encoding="latin1", names=["Sentence", "Intent"])

    yPred = []
    yTrue = []
    for i, j in complexTest.iterrows(): #i in index, j is value at row i
        pred = predict(j['Sentence'])
        intents = get_intent(pred)
        yPred.append(intents)
        yTrue.append(j['Intent'])

    print(f"1000 test model accuracy = {100*accuracy_score(yTrue, yPred)}%.\n")
    print(f"1000 test model F1 = {100*f1_score(yTrue, yPred, average='macro')}%.\n")

    yPred = []
    yTrue = []
    for i, j in complexTest.iterrows(): #i in index, j is value at row i
        pred = predict(j['Sentence'])
        intents = get_intent(pred)
        yPred.append(intents)
        yTrue.append(j['Intent'])

    print(f"Complex test model accuracy = {100*accuracy_score(yTrue, yPred)}%.\n")
    print(f"Complex test model F1 = {100*f1_score(yTrue, yPred, average='macro')}%.\n")


if __name__ == "__main__":
    compute_accuracy_and_f1_fallback_baseline()
    compute_accuracy_and_f1_model()