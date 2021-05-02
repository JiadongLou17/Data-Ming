#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:33:18 2021

@author: loujiadong
"""

import pandas as pd
import numpy as np
from nltk.tokenize import  sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
import Text_processing
from Text_processing import get_wordnet_pos, clean_tokenize_text 

import data_preparation 
from data_preparation import df_traindata_text, df_traindata_label, df_Positive, df_Negative

therold_list = [-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3]

def pos_count(X):
    pos_count = 0
    clean_words = clean_tokenize_text(X)
    upper_clean_words = [i.upper() for i in clean_words]
    df_Positive_array = np.array(df_Positive)
    for word_new in upper_clean_words:
        if word_new in df_Positive_array:
            pos_count +=1
    return pos_count

def neg_count(X):
    neg_count = 0
    clean_words = clean_tokenize_text(X)
    upper_clean_words = [i.upper() for i in clean_words]
    df_Negative_array = np.array(df_Negative)
    for word_new in upper_clean_words:
        if word_new in df_Negative_array:
            neg_count +=1
    return neg_count

def sentiment_score(X):
    return (pos_count(X)-neg_count(X))/(pos_count(X)+neg_count(X)+1)

def predict(X):
    if sentiment_score(X) > therold:
        return 1
    else:
        return -1
    
def accuracy(X):
    correct_score = 0 
    all_score = len(X)
    predict_list = [] 
    df_traindata_label_array = np.array(df_traindata_label)
    for i in X:
        predict_list.append(predict(i))
    for j in range(len(X)):
        if predict_list[j] == df_traindata_label_array[j]:
            correct_score += 1
    return correct_score/len(X)

if __name__ == "__main__":
    print(sentiment_score(df_traindata_text[0]))
    for therold in therold_list:
        print(accuracy(df_traindata_text))