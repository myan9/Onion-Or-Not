# Reference: https://www.kaggle.com/ezzaldin6/naive-bayes-classifier

import string as s
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

url = 'https://raw.githubusercontent.com/lukefeilberg/onion/master/OnionOrNot.csv'
dataset = pd.read_csv(url)
dataset.head()

x = dataset.text
y = dataset.label
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1)


def tokenization(text):
    lst = text.split()
    return lst

train_x = train_x.apply(tokenization)
test_x = test_x.apply(tokenization)


def lowercasing(lst):
    new_lst = []
    for i in lst:
        i = i.lower()
        new_lst.append(i)
    return new_lst


train_x = train_x.apply(lowercasing)
test_x = test_x.apply(lowercasing)


def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            i = i.replace(j, '')
        new_lst.append(i)
    return new_lst


train_x = train_x.apply(remove_punctuations)
test_x = test_x.apply(remove_punctuations)


def remove_numbers(lst):
    nodig_lst = []
    new_lst = []
    for i in lst:
        for j in s.digits:
            i = i.replace(j, '')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i != '':
            new_lst.append(i)
    return new_lst


train_x = train_x.apply(remove_numbers)
test_x = test_x.apply(remove_numbers)

train_x = train_x.apply(lambda x: ''.join(i+' ' for i in x))
test_x = test_x.apply(lambda x: ''.join(i+' ' for i in x))


tfidf = TfidfVectorizer()
train_1 = tfidf.fit_transform(train_x)
test_1 = tfidf.transform(test_x)

train_arr = train_1.toarray()
test_arr = test_1.toarray()


NB_MN = MultinomialNB()

def select_model(x, y, model):
    scores = cross_val_score(model, x, y, cv=5, scoring='f1')
    acc = np.mean(scores)
    return acc

select_model(train_arr, train_y, NB_MN)
NB_MN.fit(train_arr, train_y)

pred = NB_MN.predict(test_arr)
print("accuracy: ", accuracy_score(test_y, pred))
print("precision: ", precision_score(test_y, pred))
print("recall: ", recall_score(test_y, pred))
print("f1: ", f1_score(test_y, pred))
