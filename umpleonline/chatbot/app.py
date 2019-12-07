#!/usr/bin/python3
import nltk

from processresponse import process_response_baseline

from flask import Flask, escape, request
from flask_cors import CORS

app = Flask(__name__)

# Allow cross-server communication
CORS(app)

DEBUG_MODE = True
PORT_NUMBER = 8003


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    name = request.form.get("name", "World")
    user_input = request.form
    print(user_input)
    return f'Hello, {user_input}!'


@app.route("/process-input", methods=['GET', 'POST'])
def process_user_input():
    user_input = request.form.get("input")
    if DEBUG_MODE:
        return process_response_baseline(user_input)
    else:
        # TODO
        pass


def setup_deps():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('wordnet')


if __name__ == "__main__":
    setup_deps()
    app.run(debug=DEBUG_MODE, port=PORT_NUMBER)
