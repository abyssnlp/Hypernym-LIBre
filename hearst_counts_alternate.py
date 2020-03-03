#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:12:17 2019

@author: srawat
"""

# Extractions from wikipedia gonna be a tsv format file

# Extract hearst patterns and their counts or frequencies and output them to a file

# Write hypo-hypers, count to a dataframe

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

lemmatizer=WordNetLemmatizer()
stopwords=stopwords.words('english')

import re
# re.sub(r'[^a-zA-z\s]',r'','a stubborns. /657567/  . " e241234gg')
# tokenizer.tokenize('a stubborns. . " egg')

def post_process(term):
    term=re.sub(r'[^a-zA-z\s]',r'',term)
    term=re.sub("(?<=[a-z])'(?=[a-z])", "", term)
    term=term.lower()
    punct = string.punctuation
    words=term.split()
    words=[word for word in words if word not in stopwords and word not in punct]
    words=[lemmatizer.lemmatize(word) for word in words]
    return '_'.join(words)

# post_process('a stubborns. /657567/  . " e241234gg')

# def alt_extraction_to_dataframe(filename):
#     fdist=FreqDist()
#     with open(filename,'r') as f:
#         for line in f:
#             hypo,hyper=line.split('\t')
#             fdist[(hypo,hyper)]+=1
#     return fdist

# alt_extraction_to_dataframe('brown_corpus_hypernyms')
# extractions_to_dataframe('brown_corpus_hypernyms')
from collections import Counter
import pandas as pd


## test
#with open('test_hypernyms.txt','r') as f:
#    test_count=Counter()
#    for line in f:
#        hypo,hyper=line.split('\t')
#        hypo,hyper=post_process(hypo),post_process(hyper)
#        test_count[(hypo,hyper)]+=1
#test_df=pd.DataFrame(columns=['hypo','hyper','count'])
#hypos=[]
#hypers=[]
#counts=[]
#for pair,count in test_count.items():
#    hypos.append(pair[0])
#    hypers.append(pair[1])
#    counts.append(count)
#pd.DataFrame({'hypo':hypos,'hyper':hypers,'count':counts}).sort_values(by='count',ascending=False).reset_index(drop=True)

def extractions_to_dataframe(filename):
    hypo_hyper_count=Counter()
    with open(filename,'r') as f:
        for line in f:
            hypo,hyper=line.split('\t')
            hypo,hyper=post_process(hypo),post_process(hyper)
            hypo_hyper_count[(hypo,hyper)]+=1
    hypos=[]
    hypers=[]
    counts=[]
    for pair,count in hypo_hyper_count.items():
        hypos.append(pair[0])
        hypers.append(pair[1])
        counts.append(count)
    hypo_hyper_df=pd.DataFrame({'hypo':hypos,'hyper':hypers,'count':counts}).sort_values(by='count',ascending=False).reset_index(drop=True)
    hypo_hyper_df.to_csv('main_hypernym_counts.tsv',sep='\t',index=False)
    return hypo_hyper_df


import argparse
parser=argparse.ArgumentParser()

parser.add_argument('--input','-I',help='File containing Hypernyms Extracted')

args=parser.parse_args()

if args.input:
    extractions_to_dataframe(args.input)
