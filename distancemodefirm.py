#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:29:04 2017

@author: prudhvi
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:49:19 2017

@author: prudhvi
"""

import numpy as np
import pickle
import os

os.chdir('Desktop/w2v')
homolist = open('homonyms_list.txt', 'rb')
homolist = pickle.load(homolist)

homos = homolist.keys()

topic1= open('Tasa Documents/business.txt', 'rb')
topic1 = pickle.load(topic1)

topic2= open('Tasa Documents/health.txt', 'rb')
topic2 = pickle.load(topic2)

t1_sent =[]
for t in topic1:
    for te in t:
        if '[S]' in te:
            t1_sent.append(te[4:].split())
            
t2_sent =[]
for t in topic2:
    for te in t:
        if '[S]' in te:
            t2_sent.append(te[4:].split())
            
c1 = dict()
for h in homos:
    print h
    for t in t1_sent:
        if h in t:
            if h not in c1:
                c1[h] = 1
            else:
                c1[h] +=1
                  
c2 = dict()
for h in homos:
    print h
    for t in t2_sent:
        if h in t:
            if h not in c2:
                c2[h] = 1
            else:
                c2[h] +=1
                  
overlap_words=  []
for h in homos:
    if h in c1 and h in c2:
        overlap_words.append(h)

# firm - business 45 health 5
#overlap_words = ['pupil']
t1_occurences = dict()
for o in overlap_words:
    for t in t1_sent:
        if o in t:
            if o not in t1_occurences:
                t1_occurences[o] = 1
            else: 
                t1_occurences[o]+=1

t2_occurences = dict()
for o in overlap_words:
    for t in t2_sent:
        if o in t:
            if o not in t2_occurences:
                t2_occurences[o] = 1
            else: 
                t2_occurences[o]+=1

#t1h, t2h = [], []
word = 'firm'
for t in t2_sent:
    if word in t:
#        if 'business' in t:
        print t
        #t2h.append(t)

def score(base, current):
    num = 1.0 * len(np.intersect1d(base, current))
    den = len(np.union1d(base, current))
    return num/den

from scipy.spatial.distance import cosine

def cosine_score(model, words):
    sum_ = 0
    base = model.wv[word]
    for w in words:
        sum_ += cosine(base, model.wv[w])
    return sum_

import gensim

data = t2_sent + t1_sent
np.random.shuffle(data)
basemodel = gensim.models.Word2Vec(data, sg = 1, size = 100, window = 10, iter = 5, seed = 122)

basewords = dict(basemodel.most_similar(positive = ['firm'], topn = 50)).keys()

cosine_score(basemodel, basewords)

t1_second = t2_sent + t1_sent
t2_second = t1_sent + t2_sent
random = t1_sent + t2_sent
np.random.shuffle(random)

t1s, t2s, rand = [], [], []

for i in range(0, 200):

    print i
    np.random.shuffle(random)    
    randmodel = gensim.models.Word2Vec(random, sg=1, size = 100, window = 10, iter = 5, seed=  i)
    randwords = dict(randmodel.most_similar(positive = [word], topn = 50)).keys()
    rand.append(cosine_score(randmodel, randwords))
    
    t1secondmodel = gensim.models.Word2Vec(t1_second, sg=1, size = 100, window = 10, iter = 5, seed=  i)
    t1secondwords = dict(t1secondmodel.most_similar(positive = [word], topn = 50)).keys()
    t1s.append(cosine_score(t1secondmodel, t1secondwords))

    t2secondmodel = gensim.models.Word2Vec(t2_second, sg=1, size = 100, window = 10, iter = 5, seed=  i)
    t2secondwords = dict(t2secondmodel.most_similar(positive = [word], topn = 50)).keys()
    t2s.append(cosine_score(t2secondmodel, t2secondwords))

    print rand[-1], t1s[-1], t2s[-1]
    
    
import pandas as pd

df = pd.DataFrame([rand, t1s, t2s]).T
df.columns = ['rand', 't1s', 't2s']

df.to_csv('firm_cosinet1second_vs_all_200_runs_50_neigh.csv')
