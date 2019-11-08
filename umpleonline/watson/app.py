#!/usr/bin/python3
import json
import re

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
        return process_response_debug(user_input)
    else:
        # TODO
        pass


def process_response_debug(user_input):
    """
    Function used to reply with canned responses to help developers debug the app.
    It is also used as a backup if we cannot process the user input, eg for functionality not yet implemented.
    This function assumes valid input.
    """
    
    print("Processing request in debug mode")
    message_text = user_input.lower()
    words = message_text.split(' ')

    if 'create a' in message_text:  # supports a/an
        for i in range(len(words) - 2):
            if words[i] == 'create':
                # strip punctuation
                class_name = first_letter_uppercase(re.sub(r"/\s+/g", " ", re.sub(r"/[^\w\s]|_/g", "", words[i + 2])))
                return add_class(class_name)

    if "has a" in message_text:
        for i in range(len(words) - 2):
            if words[i] == 'has':
                class_name = first_letter_uppercase(words[i - 1])
                attribute_name = re.sub(r"/\s+/g", " ", re.sub(r"/[^\w\s]|_/g", "", words[i + 2]))
                return add_attribute(class_name, attribute_name)

    return "Error"


def add_class(class_name):
    return json.dumps({
        "intents": [{"intent": 'create_class'}],
        "entities": [{"value": class_name}],
        "output": {"text": [f"I created a class called {class_name}."]}
    })


def add_attribute(class_name, attribute_name):
    return json.dumps({
        "intents": [{"intent": 'add_attribute'}],
        "entities": [{"value": class_name}, {"value": attribute_name}],
        "output": [{"text": f"{class_name} now has the attribute {attribute_name}."}]
    })



def first_letter_uppercase(user_input):
    return user_input[0].upper() + user_input[1:]


if __name__ == "__main__":
    app.run(debug=DEBUG_MODE, port=PORT_NUMBER)
