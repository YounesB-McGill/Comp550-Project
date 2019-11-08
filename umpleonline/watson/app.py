#!/usr/bin/python3
from flask import Flask, escape, request
from flask_cors import CORS

app = Flask(__name__)

# Allow cross-server communication
CORS(app)

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    name = request.form.get("name", "World")
    user_input = request.form
    print(user_input)
    return f'Hello, {user_input}!'
