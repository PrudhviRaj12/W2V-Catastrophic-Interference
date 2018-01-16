# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 23:24:34 2017

@author: prudh
"""


import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#os.chdir("PyT")
from scipy.spatial.distance import cosine
os.chdir("Desktop")
data = open('w2v/Artificial_corpus_2_senses_6000.txt').read().splitlines()

ac1 = []
ac2=  []
ac3 = []
for c in data:
    if c.split()[2] == 'car' or c.split()[2] == 'truck':
        ac1.append(c)
    elif c.split()[2] == 'glass' or c.split()[2] == 'plate':
        ac2.append(c)
 #   else:
 #       ac3.append(c)

data = ac2 + ac1[len(ac1) - len(ac2)/2:]

sentences = []
for d in data:
    sentences.append(d.split())

import gensim


vocab = []
for d in data:
    for de in d.split():
        if de not in vocab:
            vocab.append(de)

#for i in range(0, 100):
    
def return_vectors(model, vocab):
    vector_dict = {}
    for v in vocab:
        #print v
        vector_dict[v] = model.wv[v]
    return vector_dict
    
def save_vectors(vector_dict, i):
    filename = open("C:/Users/prudh/W2V-Catastrophic-Interference/GensimImplementation/Vecs 2/Vehicles_Dinnerware_News 3 Senses 3 Iter/vehi_dinner_news_vectors_" +str(i) + ".pkl", "wb")
    pickle.dump(vector_dict, filename)

from collections import defaultdict

dic = defaultdict(dict)    
for i in range(0, 10000):
    #np.random.shuffle(sentences)
    print i
    model = gensim.models.Word2Vec(sentences, sg = 1, size = 300, window=2, iter = 5, seed = i)
    vectors = return_vectors(model, vocab)
    dic[i] = vectors    

import copy

full = copy.deepcopy(dic)

query_word=  'break'
#check_word = ['car', 'truck', 'glass', 'plate', 'news', 'story']
check_word = ['car', 'truck', 'glass', 'plate']

from collections import defaultdict

dic = defaultdict(dict)
for v in range(0, 10000):
    first = full[v]
    for c in check_word:
        dic[v][c] = cosine(first[query_word], first[c])

import pandas as pd
dframe = pd.DataFrame(dic).T

dframe['Vehicles'] = (dframe['car'] + dframe['truck'])/2
dframe['Dinnerware'] = (dframe['glass'] + dframe['plate'])/2
#dframe['News'] = (dframe['news'] + dframe['story'])/2


dframe['closer to vehicles'] = (dframe['Vehicles'] < dframe['Dinnerware']) #& (dframe['Vehicles'] < dframe['News']) 
dframe['closer to dinnerware'] = (dframe['Dinnerware'] < dframe['Vehicles'])# & (dframe['Dinnerware'] < dframe['News']) 

#dframe['closer to vehicles'] = (dframe['Vehicles'] < dframe['Dinnerware']) & (dframe['Vehicles'] < dframe['News']) 
#dframe['closer to dinnerware'] = (dframe['Dinnerware'] < dframe['Vehicles']) & (dframe['Dinnerware'] < dframe['News']) 
#dframe['closer to news'] = (dframe['News'] < dframe['Dinnerware']) & (dframe['News'] < dframe['Vehicles']) 

print "Vehicles Occuring First: " + str(sum(dframe['closer to vehicles']))
print "Dinnerware Occuring First: " + str(sum(dframe['closer to dinnerware']))
#print "News Occuring First: " + str(sum(dframe['closer to news']))

print "Proportion of Vehicles Occuring First: " + str(sum(dframe['closer to vehicles'])/10000.0)
print "Proportion of Dinnerware Occuring First: " + str(sum(dframe['closer to dinnerware'])/10000.0)
#print "Phroportion of News Occuring First: " + str(sum(dframe['closer to news'])/10000.0)

dframe.to_csv('VehicleDinnerware 10000 Runs Unbal dinner_vehi1.csv')