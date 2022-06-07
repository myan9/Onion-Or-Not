# Reference: https://www.kaggle.com/sean49/onion-or-not-a-binary-classification-with-keras/notebook
#            https://www.kaggle.com/salmanhiro/onn-onion-neural-network-with-lstm-85-accuracy

import re
import contractions
import pandas as pd
import en_core_web_sm

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional, GlobalMaxPool1D

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/lukefeilberg/onion/master/OnionOrNot.csv'
dataset = pd.read_csv(url)

# show data
dataset.head()
dataset.info()

# fix contraction
dataset['text'] = dataset['text'].apply(lambda x: contractions.fix(x))

# remove puntuation
dataset['text'] = dataset['text'].str.replace('[^a-zA-Z]', ' ', regex=True)

# convert to lowercase
dataset['text'] = dataset['text'].str.lower()

# show the precrocessed data
dataset.head(5)

# lemmatization
sp = en_core_web_sm.load()


def lemma(input_str):
    s = sp(input_str)

    input_list = []
    for word in s:
        w = word.lemma_
        input_list.append(w)

    output = ' '.join(input_list)
    return output


dataset['text'] = dataset['text'].apply(lambda x: lemma(x))

# show the precrocessed data
dataset.head(5)

# tokenization
tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(dataset['text'].values)

X = tokenizer.texts_to_sequences(dataset['text'].values)
X = pad_sequences(X)

y = dataset['label']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# build the model
model = Sequential()

model.add(Embedding(10000, 128, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(GlobalMaxPool1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.summary()

# train the model
earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit(x_train, y_train, epochs=1000, batch_size=32,
                    validation_data=(x_test, y_test), callbacks=[earlystop])

# evaluation
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5

print("accuracy: ", accuracy_score(y_pred, y_test))
print("precision: ",  precision_score(y_pred, y_test))
print("recall: ", recall_score(y_pred, y_test))
print("f1 score: ", f1_score(y_pred, y_test))
