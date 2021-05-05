#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:21:52 2021

@author: loujiadong
"""
import numpy as np
import xgboost
# First XGBoost model for Pima Indians dataset
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
 
Count_vec = CountVectorizer( stop_words='english', ngram_range=(1, 2))# #sparse=False意思是不产生稀疏矩阵
# vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
import data_preparation 
from data_preparation import df_traindata_text, df_traindata_label

import Text_processing
from Text_processing import get_wordnet_pos, clean_tokenize_text 
# split data into X and y
X = df_traindata_text
X = np.array(X)
Y = df_traindata_label
Y = np.array(Y)
# split data into train and test sets
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
X_train = np.array(X_train)
X_train = Count_vec.fit_transform(X_train)#.to_dict(orient='record'
X_test = Count_vec.transform(X_test)# .to_dict(orient='record'


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value,4) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Accuracy: 79.15%