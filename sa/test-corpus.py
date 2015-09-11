# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:36:55 2015

@author: VAIO
"""
from collections import defaultdict
from sklearn.pipeline import make_pipeline, make_union
from corpus import iter_corpus, iter_test_corpus
from transformations import (ExtractText, ExtractAuthor,ExtractDate,EncodingText)
import csv
from settings import DATA_PATH

def target(phrases):
    return [datapoint.rating for datapoint in phrases]
    
phrases = list(iter_corpus())

vocabulary=[]
with open(DATA_PATH + '/vocabulary',encoding='utf-8') as f:
    rd=csv.reader(f)
    for line in rd:
        vocabulary.append(line[0])
          
pipeline1 = [ExtractText()]
pipeline1.append(EncodingText(vocabulary))
pipeline=make_pipeline(*pipeline1)


y = target(phrases)


Z = pipeline.fit_transform(phrases, y)

print(Z[0])


"""
结果是一个list，list的每个元素是语料库映射成数字的句子
"""