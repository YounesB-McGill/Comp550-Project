#!/usr/bin/python3

"""
Comp550 Project - Fall 2019

Modeling Software Using Natural Language

Group 48: Younes Boubekeur, Anmoljeet Gill, Trung Vuong Thien
"""

import nltk
from flask import Flask, escape, request
from flask_cors import CORS

from model import predict, getIntent, keyIntent
from processresponse import process_response_baseline, process_response_model


app = Flask(__name__)

# Allow cross-server communication
CORS(app)

USE_BASELINE = False
DEBUG_MODE = True
PORT_NUMBER = 8003

MODEL_FILE = "model/modelLSTM.h5"


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    name = request.form.get("name", "World")
    user_input = request.form
    print(user_input)
    return f'Hello, {user_input}!'


@app.route("/process-input", methods=['GET', 'POST'])
def process_user_input():
    user_input = request.form.get("input")
    if USE_BASELINE:
        return process_response_baseline(user_input)
    else:
        return process_response_model(user_input)


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('words')


def init_model():
    prediction = predict("Send this sentence to model to initialize it.")
    print(f"Model called, prediction is:\n{getIntent(prediction, keyIntent)}")


if __name__ == "__main__":
    setup_deps()
    init_model()
    app.run(debug=DEBUG_MODE, port=PORT_NUMBER, threaded=False)
