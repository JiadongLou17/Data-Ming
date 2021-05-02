#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:08:55 2021

@author: loujiadong
"""

from nltk.tokenize import  sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

def get_wordnet_pos(treebank_tag):
    # transfer nltk pos tag to wordnet postag
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

def clean_tokenize_text(text,remove_stopwords=True):# 是否移除
 
    """
    Clean and tokenize text into words
    """
    #Step1:tokenize
    words = word_tokenize(text)
    #Step2:pos tagging
    word_tags = nltk.pos_tag(words)

    wnl = WordNetLemmatizer()
    filtered_words=[]
    
    for word_tag in word_tags:   
        wornet_tag = get_wordnet_pos(word_tag[1])
         #step3: Lemmatization or stemming
        word_lemma = wnl.lemmatize(word_tag[0],pos=wornet_tag)
        #step4: normalize case
        low_word_lemma = word_lemma.lower()
         #step5: remove stop words
        if remove_stopwords and low_word_lemma in stop_words: 
            continue
        filtered_words.append(low_word_lemma)
    # count word fequency
    return filtered_words


import data_preparation 
from data_preparation import df_traindata_text


if __name__ == "__main__":
    clean_words = clean_tokenize_text(df_traindata_text[0])
    word_count = nltk.FreqDist(clean_words)
    word_count.tabulate()#print the word freq
    
    print("="*20)
    
    clean_words = clean_tokenize_text(df_traindata_text[0],False)#without removing stopwords
    word_count = nltk.FreqDist(clean_words)
    word_count.tabulate()