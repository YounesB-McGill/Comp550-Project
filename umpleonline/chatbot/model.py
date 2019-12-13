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


# This file was originally written as a Jupyter notebook.

MODEL_FILE = "model/modelLSTM.h5"
CSV_FILE = "data/dataset.csv"
TEST_SIZE = 0.15

dataSet = pd.read_csv(CSV_FILE, encoding="latin1", names=["Sentence", "Intent"])
intent = dataSet["Intent"]
keyIntent = list(set(intent))

intent = dataSet["Intent"]
keyIntent = list(set(intent))
sentences = list(dataSet["Sentence"])


def clean(sentences):
    wordList = []
    for i in sentences:
        subbedSentence = re.sub(r'[^ a-z A-Z 0-9]', " ", i)
        words = word_tokenize(subbedSentence)
        wordList.append([j.lower() for j in words]) 
    return wordList


cleanedSentences = clean(sentences)


def tokens(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token


wordTokens = tokens(cleanedSentences)


vocabSize = len(wordTokens.word_index) + 1


"""
Vocab Size = 365 and Maximum length = 8
"""

def maxLength(words):
    return(len(max(words, key = len)))


maxLengths = maxLength(cleanedSentences)


def encoder(tokens, words):
    return(tokens.texts_to_sequences(words))


def padder(words, maxLen):
    return(pad_sequences(words, maxlen = maxLengths, padding = "post"))


def oneHotEncoder(encode):
    oneHot = OneHotEncoder(sparse = False)
    return(oneHot.fit_transform(encode))


def create_model(vocabSizez, maxLen, input_shape):

    model = Sequential()
    model.add(Embedding(vocabSizez, 128))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = "softmax"))
  
    return model


model = load_model(MODEL_FILE)


def predict(text):
    cleanedSentence = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    testWord = word_tokenize(cleanedSentence)
    testWord = [w.lower() for w in testWord]
    testTokens = wordTokens.texts_to_sequences(testWord)
    
    print(testWord)
    if [] in testTokens: #Check for unknown words
        testTokens = list(filter(None, testTokens))
    
    testTokens = np.array(testTokens).reshape(1, len(testTokens))
    x = padder(testTokens, maxLengths)
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

    encodedSentences = encoder(wordTokens, cleanedSentences)

    paddedSentences = padder(encodedSentences, maxLengths)

    intentTokens = tokens(keyIntent, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
    tokenOutput = encoder(intentTokens, intent)
    tokenOutput = np.array(tokenOutput).reshape(len(tokenOutput), 1)
    oneHotOutput = oneHotEncoder(tokenOutput)

    xTrain, xValid, yTrain, yValid = train_test_split(paddedSentences, oneHotOutput, shuffle=True, test_size=TEST_SIZE)

    input_shape = xTrain.shape
    print(input_shape)
    model = create_model(vocabSize, maxLengths, input_shape)
    model.summary()
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()

    modelFileName = 'modelLSTM.h5'

    # This operation take a long time (~2h on Anmojeet's machine)
    checkpoint = ModelCheckpoint(modelFileName, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = model.fit(xTrain, yTrain, epochs = 150, batch_size = 32, validation_data = (xValid, yValid), callbacks = [checkpoint])
    model.save(modelFileName)

    """
    Sample training output:

    (2121, 8)
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_21 (Embedding)     (None, None, 128)         46720     
    _________________________________________________________________
    bidirectional_5 (Bidirection (None, 256)               263168    
    _________________________________________________________________
    dense_82 (Dense)             (None, 64)                16448     
    _________________________________________________________________
    dropout_60 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    dense_83 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    dropout_61 (Dropout)         (None, 32)                0         
    _________________________________________________________________
    dense_84 (Dense)             (None, 5)                 165       
    =================================================================
    Total params: 328,581
    Trainable params: 328,581
    Non-trainable params: 0
    _________________________________________________________________
    Train on 2121 samples, validate on 375 samples
    Epoch 1/150
    2121/2121 [==============================] - 6s 3ms/step - loss: 1.2152 - acc: 0.4781 - val_loss: 0.3357 - val_acc: 0.9813

    Epoch 00001: val_loss improved from inf to 0.33572, saving model to modelLSTM.h5
    Epoch 2/150
    2121/2121 [==============================] - 2s 774us/step - loss: 0.2965 - acc: 0.8878 - val_loss: 0.0029 - val_acc: 1.0000

    Epoch 00002: val_loss improved from 0.33572 to 0.00293, saving model to modelLSTM.h5
    Epoch 3/150
    2121/2121 [==============================] - 1s 575us/step - loss: 0.1024 - acc: 0.9642 - val_loss: 1.8908e-04 - val_acc: 1.0000

    Epoch 00003: val_loss improved from 0.00293 to 0.00019, saving model to modelLSTM.h5
    Epoch 4/150
    2121/2121 [==============================] - 1s 582us/step - loss: 0.0659 - acc: 0.9755 - val_loss: 3.6352e-05 - val_acc: 1.0000

    [...]

    Epoch 00150: val_loss did not improve from 0.00000
    """

    text = "wire is vessel"
    pred = predict(text)
    print(getIntent(pred, keyIntent))

    """
    Sample output:
    
    ['wire', 'is', 'vessel']
    create_inheritance has confidence = 1.0
    add_attribute has confidence = 0.0
    create_composition has confidence = 0.0
    create_association has confidence = 0.0
    add_class has confidence = 0.0
    None
    """
