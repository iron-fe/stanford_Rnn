# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:14:11 2015

@author: VAIO
"""
import os
from corpus import cleanText, wordSeg
import numpy as np
from settings import DATA_PATH
import csv

def buildVocab():
    source = "/Users/VAIO/doc2vec/ChnSentiCorp_htl_ba_2000/"
    pos_reviews = []
    neg_reviews = []
    for root, dirs, files in os.walk( source ):
        if root.find('neg') != -1:
            for texts in files:
                if texts != '.DS_Store':
                    with open(root + '/' + texts,'r',encoding='utf-8') as infile:
                        neg_reviews.append(infile.read())
        if root.find('pos') != -1:
            for texts in files:
                if texts != '.DS_Store':
                    with open(root + '/' + texts,'r',encoding='utf-8') as infile:
                        pos_reviews.append(infile.read())
    phrases = np.concatenate((pos_reviews, neg_reviews))
    phrases = cleanText(phrases)
    phrases = wordSeg(phrases)
    vocab = [c.split(" ") for c in phrases]
    vocab = [j for i in vocab for j in i]
    vocab = list(set(vocab))
    with open(DATA_PATH +'/vocabulary_cn','w',encoding='utf-8') as f:
            wrt=csv.writer(f)
            for i in range(len(vocab)):
#                datapoint=[phrases[i].id, phrases[i].date, phrases[i].content, phrases[i].rating, pred[i]]
                word=[vocab[i]]
                wrt.writerow(word)    
    
buildVocab()