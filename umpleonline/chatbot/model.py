#!/usr/bin/python3

import re

import numpy as np
import pandas as pd
import nltk

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


TEST_SIZE = 0.15

def clean(sentences):
    wordList = []
    for i in sentences:
        subbedSentence = re.sub(r'[^ a-z A-Z 0-9]', " ", i)
        words = word_tokenize(subbedSentence)
        wordList.append([j.lower() for j in words]) 
    return wordList


def tokens(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token


def maxLength(words):
    return(len(max(words, key = len)))


def encoder(tokens, words):
    return(tokens.texts_to_sequences(words))


def padder(words, maxLen):
    return(pad_sequences(words, maxlen = maxLengthz, padding = "post"))


def oneHotEncoder(encode):
    oneHot = OneHotEncoder(sparse = False)
    return(oneHot.fit_transform(encode))


def create_model(vocabSizez, maxLen, input_shape):

    model = Sequential()
    model.add(Embedding(vocabSizez, 128))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = "softmax"))
  
    return model


def predict(text):
    cleanedSentence = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    testWord = word_tokenize(cleanedSentence)
    testWord = [w.lower() for w in testWord]
    testTokens = wordTokens.texts_to_sequences(testWord)
    
    print(testWord)
    if [] in testTokens: #Check for unknown words
        testTokens = list(filter(None, testTokens))
    
    testTokens = np.array(testTokens).reshape(1, len(testTokens))
    x = padder(testTokens, maxLengthz)
    pred = model.predict_proba(x)
  
    return pred


def getIntent(predicts, intents):
    prediction = predicts[0]
 
    intents = np.array(intents)
    ids = np.argsort(-prediction)
    intents = intents[ids]
    predictions = -np.sort(-prediction)
 
    for i in range(predicts.shape[1]):
        print("%s has confidence = %s" % (intents[i], (predictions[i])))



if __name__ == "__main__":
    dataSet = pd.read_csv('merged.csv', encoding = "latin1", names = ["Sentence", "Intent"])
    print(dataSet.head())
    intent = dataSet["Intent"]
    keyIntent = list(set(intent))
    sentences = list(dataSet["Sentence"])

    cleanedSetences = clean(sentences)
    wordTokens = tokens(cleanedSetences)

    vocabSize = len(wordTokens.word_index) + 1
    maxLengthz = maxLength(cleanedSetences)

    encodedSentences = encoder(wordTokens, cleanedSetences)

    paddedSentences = padder(encodedSentences, maxLengthz)

    intentTokens = tokens(keyIntent, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
    tokenOutput = encoder(intentTokens, intent)
    tokenOutput = np.array(tokenOutput).reshape(len(tokenOutput), 1)
    oneHotOutput = oneHotEncoder(tokenOutput)

    xTrain, xValid, yTrain, yValid = train_test_split(paddedSentences, oneHotOutput, shuffle=True, test_size=TEST_SIZE)

    input_shape = xTrain.shape
    print(input_shape)
    model = create_model(vocabSize, maxLengthz, input_shape)
    model.summary()
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    modelFileName = 'modelLSTM.h5'

    #These take a long time
    #checkpoint = ModelCheckpoint(modelFileName, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #hist = model.fit(xTrain, yTrain, epochs = 150, batch_size = 32, validation_data = (xValid, yValid), callbacks = [checkpoint])
    #model.save(modelFileName)

    model = load_model('modelLSTM.h5')

    text = "wire is vessel"
    pred = predict(text)
    print(getIntent(pred, keyIntent))

    """
    ['wire', 'is', 'vessel']
    create_inheritance has confidence = 1.0
    add_attribute has confidence = 0.0
    create_composition has confidence = 0.0
    create_association has confidence = 0.0
    add_class has confidence = 0.0
    None
    """
