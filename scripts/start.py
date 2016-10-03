# -*- coding: utf-8 -*-
"""
Created on Thu May  5 01:46:07 2016

@author: sidvash
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
import re

stop_words = set(stopwords.words("english"))

#Only for spyder -removes run time warning messages
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


def lower_case(string):
    return string.lower()
    
    
def string_replace(s):
    if isinstance(s, str):
       
       #Seperators        
        s = re.sub(r"([a-zA-Z])\.([a-zA-Z)])", r"\1 \2", s) #sep '.' b/w letters
        s=re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", s) #sep alpha anumeric
        s=re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", s)
        s=re.sub(r"([a-z])([A-Z])", r"\1 \2",s) #sep lowercase and uppercase letter
           
        #removes any , adjacent to an alphabet
        s = re.sub(r"\,([a-zA-Z])", r" \1", s) 
        s = re.sub(r"([a-zA-Z])\,", r"\1 ", s) 
        
        s = re.sub(r"([a-zA-Z])\/([a-zA-Z])", r"\1 \2", s)  #sep '/' b/w letters
        s = re.sub(r"([a-zA-Z])\\([a-zA-Z])", r"\1 \2", s)  #sep '\' b/w letters
        s=re.sub(r"([a-zA-Z])(\.) ", r"\1 ", s) #removes dot after any alphabet
        s=re.sub(r"([0-9])\,([0-9])", r"\1\2", s) #removing commas in b/w numbers
        s=re.sub(r"[()]", r" ", s) #removes open and close brackets
        
        
        #replacements
        s=s.replace("-", " ")
        s=s.replace("*", " ")
        s=s.replace("#", " ")
        s=s.replace(";", " ")
        s=s.replace("$", " ")
        s=s.replace("%", " ")
        s=s.replace(",", " ")
        s=s.replace(".", " ")
        s=s.replace("/", " ")
        s=s.replace("\\", " ")
        
        #last substitution to save space
        s=re.sub(r"  ", r" ", s) #substitutes double space to single
        return s
    else:
        return "null"    

def remove_sw(string):
    words = word_tokenize(string)
    filtered_sentence = []
    for x in words:
        if x not in stop_words:
            filtered_sentence.append(x)
    final_sentence = " ".join(filtered_sentence)
    return final_sentence
    
def count_word(word, string):
    count = 0
    string1 = word_tokenize(string)
    for x in string1:
        if x == word:
            count = count+1
    return count
    
def count_unique_word(string):
    words = word_tokenize(string)
    unique_words = []    
    for x in words:
        if x not in unique_words:
            unique_words.append(x)
            
    return len(unique_words)
    
def unique_words(string):
    words = word_tokenize(string)
    unique_words = []    
    for x in words:
        if x not in unique_words:
            unique_words.append(x)
            
    return ' '.join(unique_words)


def freq_word_dict(string):
    words = word_tokenize(string)
    wordfreq = {}
    for x in words:
        if x not in wordfreq:
            wordfreq[x] = 0
        wordfreq[x] += 1
    return wordfreq
            
    
    
df_train = pd.read_csv('/home/sidvash/kaggle_2016/sentiment_movies/raw_data/train.tsv', sep = '\t')
df_test = pd.read_csv('/home/sidvash/kaggle_2016/sentiment_movies/raw_data/test.tsv', sep = '\t')
df_all = pd.concat((df_train, df_test), axis=0)


num_train = df_train.shape[0]
num_test=df_test.shape[0]

#Pre-process
df_all['filtered_phrase'] = df_all['Phrase'].map(lambda x: lower_case(x))
df_all['filtered_phrase'] = df_all['filtered_phrase'].map(lambda x: string_replace(x))
#df_all['filtered_phrase'] = df_all['filtered_phrase'].map(lambda x: remove_sw(x))



#****************   Multinomial Naive Bayes   ***************
#*** Calculating Priors ***
s_class = df_train.Sentiment.unique()  #array of classes

#no of Examples in each class
count_no_class ={ x: df_train[df_train.Sentiment == x].shape[0] for x in s_class}

#Dictionary of priors
p_class = {x: count_no_class[x]/num_train for x in count_no_class}

#*** Text dictionary of each class
text_class = {x: ' '.join(df_all[df_all.Sentiment == x].filtered_phrase) for x in s_class}

#no of words in each class
count_word_class = {x: len(word_tokenize(text_class[x])) for x in s_class}

#Vocabulary of training set
train_vocab = count_unique_word(' '.join(df_all["filtered_phrase"][:num_train]))

#Freq dictionaries:
categ_freq_word = { x: freq_word_dict(text_class[x]) for x in s_class}


def freq_word_in_categ(word, categ):
    if word in categ_freq_word[categ]:
        freq = categ_freq_word[categ][word]
    else:
        freq = 0
    return freq
    
    
def log_prob_string(string, categ,vocab):
    words = word_tokenize(string)
    log_prob = 0
    for x in words:
        log_prob += math.log10(freq_word_in_categ(x,categ)+1) - math.log10(count_word_class[categ]+train_vocab)
    return log_prob + math.log10(p_class[categ])

#Predict probabilities for each class        
def return_categ(string):
    prob = {x: log_prob_string(string,x,train_vocab) for x in s_class}
    return max(prob, key=prob.get)
    
df_train = df_all[:num_train]
df_test = df_all[num_train:]
 
df_train['preds'] = df_train['filtered_phrase'].map(lambda x: return_categ(x))  
train_accuracy = (df_train[df_train.preds == df_train.Sentiment].shape[0]) / (df_train.shape[0])
 
df_test['preds'] = df_test['filtered_phrase'].map(lambda x: return_categ(x))      
df_submit = df_test[['PhraseId', 'preds']].rename(columns = {'preds':'Sentiment'})    
df_submit.to_csv('/home/sidvash/kaggle_2016/sentiment_movies/submissions/naive_bayes_incl_sw.csv', index=False)        