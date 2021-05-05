#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:50:50 2021

@author: loujiadong
"""

import pandas as pd
import numpy as np

df_traindata = pd.read_excel('train.xlsx')
df_traindata.dropna(inplace=True)
#print(df_traindata.head())
#len(df_traindata['text'])

# select the text and label
df_traindata_text = df_traindata['text']
df_traindata_label = df_traindata['label']

# load different datasets of dif words
df_Negative = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='Negative', header=None)
df_Positive = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='Positive', header=None)
df_Uncertainty = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='Uncertainty', header=None)
df_Litigious = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='Litigious', header=None)
df_StrongModal = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='StrongModal', header=None)
df_WeakModal = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='WeakModal', header=None)
df_Constraining = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', 
                   sheet_name='Constraining', header=None)