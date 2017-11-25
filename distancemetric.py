#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:17:47 2017

@author: prudhvi
"""

import numpy as np
import pickle
import os

os.chdir('Desktop/w2v')
homolist = open('homonyms_list.txt', 'rb')
homolist = pickle.load(homolist)

homos = homolist.keys()

topic1= open('Tasa Documents/science.txt', 'rb')
topic1 = pickle.load(topic1)

topic2= open('Tasa Documents/socialstudies.txt', 'rb')
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

overlap_words = ['pupil']
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

word = 'pupil'
for t in t1_sent:
    if word in t:
        print t


import gensim

np.random.shuffle(t2_sent) 
np.random.shuffle(t1_sent)

total = t1_sent + t2_sent
np.random.shuffle(total)


baserandommodelt1 = gensim.models.Word2Vec(t1_sent, sg = 1, size = 300, window=10, iter = 5, seed = 122)
basewords10t1 = baserandommodelt1.most_similar(positive = [word], topn = 100)

baserandommodelt2 = gensim.models.Word2Vec(t2_sent, sg = 1, size = 300, window=10, iter = 5, seed = 122)
basewords10t2 = baserandommodelt2.most_similar(positive = [word], topn = 100)

from scipy.spatial.distance import cosine

wordst1 = [b[0] for b in basewords10t1]
wordst2 = [b[0] for b in basewords10t2]

t1_dist, t2_dist = [], []
for i in range(0, 100):
    print i
    np.random.shuffle(total)    
    model = gensim.models.Word2Vec(total, sg = 1, size = 300, window = 10, iter = 5, seed = i)
    
    d1 = 0
    weights = np.arange(0, 1, 0.01)
    for i in range(0, len(wordst1)):
        d1 += weights[-1-i] * cosine(model.wv[word], model.wv[wordst1[i]])
        
    d2 = 0
    weights = np.arange(0, 1, 0.01)
    for i in range(0, len(wordst2)):
        d2 += weights[-1-i] * cosine(model.wv[word], model.wv[wordst2[i]])

    t1_dist.append(d1)
    t2_dist.append(d2)

    print t1_dist[-1], t2_dist[-1]

t1_dist1, t2_dist1 = [], []
total = t1_sent + t2_sent

for i in range(0, 100):
    print i,'t1second'
    #np.random.shuffle(total)    
    model = gensim.models.Word2Vec(total, sg = 1, size = 300, window = 10, iter = 5, seed = i)
    
    d1 = 0
    weights = np.arange(0, 1, 0.01)
    for i in range(0, len(wordst1)):
        d1 += weights[-1-i] * cosine(model.wv[word], model.wv[wordst1[i]])
        
    d2 = 0
    weights = np.arange(0, 1, 0.01)
    for i in range(0, len(wordst2)):
        d2 += weights[-1-i] * cosine(model.wv[word], model.wv[wordst2[i]])

    t1_dist1.append(d1)
    t2_dist1.append(d2)

    print t1_dist1[-1], t2_dist1[-1]

t1_dist2, t2_dist2 = [], []
total = t2_sent + t1_sent

for i in range(0, 100):
    print i,'t1second'
    #np.random.shuffle(total)    
    model = gensim.models.Word2Vec(total, sg = 1, size = 300, window = 10, iter = 5, seed = i)
    
    d1 = 0
    weights = np.arange(0, 1, 0.01)
    for i in range(0, len(wordst1)):
        d1 += weights[-1-i] * cosine(model.wv[word], model.wv[wordst1[i]])
        
    d2 = 0
    weights = np.arange(0, 1, 0.01)
    for i in range(0, len(wordst2)):
        d2 += weights[-1-i] * cosine(model.wv[word], model.wv[wordst2[i]])

    t1_dist2.append(d1)
    t2_dist2.append(d2)

    print t1_dist2[-1], t2_dist2[-1]






w1 = 'iris'
w2 = 'diaphragm'd
w4 = 'grades'

t1w1, t1w2, t2w1, t2w2 = [], [], [], []

from scipy.spatial.distance import cosine

for i in range(0, 100):
    print i
    #np.random.shuffle(total)
    model = gensim.models.Word2Vec(total, sg = 1, size = 100, window = 10, iter= 5, seed = i)
    a = cosine(model.wv[word], model.wv[w1])
    b = cosine(model.wv[word], model.wv[w2])
    c = cosine(model.wv[word], model.wv[w3])
    d = cosine(model.wv[word], model.wv[w4])
    t1w1.append(a)
    t1w2.append(b)
    t2w1.append(c)
    t2w2.append(d)
    
    print t1w1[-1], t1w2[-1], t2w1[-1], t2w2[-1]
    
    
import pandas as pd
df = pd.DataFrame([t1w1, t1w2, t2w1, t2w2]).T
df.columns = ['iris', 'diaphragm', 'tutoring', 'grades']

df.to_csv("t1second_cosine_4_words.csv")

df = pd.read_csv("t1second_cosine_4_words.csv")

x = np.arange(0, 100, 1)

import matplotlib.pyplot as plt
plt.plot(x, df['iris'], color = 'red')
plt.plot(x, np.repeat((np.mean(df['iris']) + np.mean(df['diaphragm']))/2, 100), color = 'red')
plt.plot(x, df['diaphragm'], color = 'red')

plt.plot(x, df['tutoring'], color = 'blue')
plt.plot(x, np.repeat((np.mean(df['tutoring']) + np.mean(df['grades']))/2, 100), color = 'blue')
plt.plot(x, df['grades'], color = 'blue')


import pandas as pd
df = pd.DataFrame([t1_dist, t2_dist, t1_dist1, t2_dist1, t1_dist2, t2_dist2]).T
df.columns = ['random t1', 'random t2', 't2second t1', 't2second t2', 't1second t1', 't1second t2']

df.to_csv('weight_cosine.csv')
import matplotlib.pyplot as plt
x = np.arange(0, 100, 1)

plt.plot(x, t1_dist2, color = 'red')
plt.plot(x, t2_dist2, color = 'blue')
plt.plot(x, np.repeat(np.mean(t1_dist2), 100), color = 'red')
plt.plot(x, np.repeat(np.mean(t2_dist2), 100), color = 'blue')




plt.savefig('rand10.png')