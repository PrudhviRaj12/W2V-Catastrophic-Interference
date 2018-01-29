#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:48:16 2018

@author: prudhvi
"""

import numpy as np
import pickle
import os

os.chdir('Desktop/w2v')
homolist = open('homonyms_list.txt', 'rb')
homolist = pickle.load(homolist)

homos = homolist.keys()

topic1= open('Tasa Documents/socialstudies.txt', 'rb')
topic1 = pickle.load(topic1)

topic2= open('Tasa Documents/science.txt', 'rb')
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

w1 = 'pupil'

def score(base, current):
    num = 1.0 * len(np.intersect1d(base, current))
    den = len(np.union1d(base, current))
    return num/den

import gensim


np.random.shuffle(t1_sent)
topic1_model = gensim.models.Word2Vec(t1_sent, sg = 1, size = 300, window = 5, iter = 5, seed = 122)
np.random.shuffle(t2_sent)
topic2_model = gensim.models.Word2Vec(t2_sent, sg = 1, size = 300, window = 5, iter = 5, seed = 122)

topic1_words = topic1_model.most_similar(positive = [w1], topn = 100)
topic2_words = topic2_model.most_similar(positive = [w1], topn = 100)

topic1_words = dict(topic1_words).keys()
topic2_words = dict(topic2_words).keys()


random = t1_sent + t2_sent
np.random.shuffle(random)

random_model = gensim.models.Word2Vec(random, sg = 1, size = 300, window = 5, iter = 5, seed = 1)
random_words = dict(random_model.most_similar(positive = [w1], topn = 200)).keys()

close_to_t1_rand = score(topic1_words, random_words)
close_to_t2_rand = score(topic2_words, random_words)

print close_to_t1_rand, close_to_t2_rand

topic1second = t2_sent + t1_sent
topic1second_model = gensim.models.Word2Vec(topic1second, sg = 1, size = 300, window = 5, iter = 5, seed = 1)
topic1second_words = dict(topic1second_model.most_similar(positive = [w1], topn = 200)).keys()

close_to_t1_t1s = score(topic1_words, topic1second_words)
close_to_t2_t1s = score(topic2_words, topic1second_words)

print close_to_t1_t1s, close_to_t2_t1s

topic2second = t1_sent + t2_sent
topic2second_model = gensim.models.Word2Vec(topic2second, sg = 1, size = 300, window = 5, iter = 5, seed = 1)
topic2second_words = dict(topic2second_model.most_similar(positive = [w1], topn = 200)).keys()

close_to_t1_t2s = score(topic1_words, topic2second_words)
close_to_t2_t2s = score(topic2_words, topic2second_words)

print close_to_t1_t2s, close_to_t2_t2s


