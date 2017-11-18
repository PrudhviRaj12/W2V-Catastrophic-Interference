#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:52:58 2017

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

total = t2_sent + t1_sent
np.random.shuffle(total)


baserandommodelt1 = gensim.models.Word2Vec(t1_sent, sg = 1, size = 100, window=10, iter = 5, seed = 122)
basewords10t1 = baserandommodelt1.most_similar(positive = [word], topn = 10)

baserandommodelt2 = gensim.models.Word2Vec(t2_sent, sg = 1, size = 100, window=10, iter = 5, seed = 122)
basewords10t2 = baserandommodelt2.most_similar(positive = [word], topn = 10)

t1w = ['iris', 'diaphragm']
t2w = ['tutoring', 'grades']

#t2model = gensim.models.Word2Vec(t2_sent, sg = 1, size = 100, window=10, iter = 5)
'''
def return_vectors(model, vocab):
    vector_dict = {}
    for v in vocab:
        if v in model:
        #print v
            vector_dict[v] = model.wv[v]
    return vector_dict

t1vocab = []
for t in t1_sent:
    for te in t:
        if te not in t1vocab:
            t1vocab.append(te)

t2vocab = []
for t in t2_sent:
    for te in t:
        if te not in t2vocab:
            t2vocab.append(te)

t1words =  dict(t1model.most_similar_cosmul(positive = [word], topn = 20)).keys()
t2words = dict(t2model.most_similar_cosmul(positive = [word], topn = 20)).keys()


total_vocab = t1vocab + t2vocab
'''

baserandomwords250 =  dict(baserandommodel.most_similar(positive = [word], topn = 250)).keys()
baserandomwords500 =  dict(baserandommodel.most_similar(positive = [word], topn = 500)).keys()
baserandomwords1000 =  dict(baserandommodel.most_similar(positive = [word], topn = 1000)).keys()

#baserandommodel.most_similar(positive = [word], topn = 50)
t1250, tr250, t2250 = [], [], []
t1500, tr500, t2500 = [], [], []
t11000, tr1000, t21000 = [], [], []

t1second = t2_sent + t1_sent
t2second = t1_sent + t2_sent
randomorder = t1_sent + t2_sent
from scipy.spatial.distance import cosine

def score(base, current):
    num = 1.0 * len(np.intersect1d(base, current))
    den = len(np.union1d(base, current))
    return num/den

for i in range(0, 100):
    print i
    #print "T2 should be greater than T1"
    #print "T1 should be greater than T2"
   # total = t2_sent + t1_sent
   # total_vocab = t1vocab + t2vocab
    #np.random.shuffle(total)
    t1secondmodel = gensim.models.Word2Vec(t1second, sg = 1, size = 100, window=10, iter = 5, seed = i)
    t1secondwords250=  dict(t1secondmodel.most_similar(positive = [word], topn = 250)).keys()
    t1secondwords500=  dict(t1secondmodel.most_similar(positive = [word], topn = 500)).keys()
    t1secondwords1000=  dict(t1secondmodel.most_similar(positive = [word], topn = 1000)).keys()
    
    
    #for j in range(10):
    np.random.shuffle(randomorder)
    
    randommodel = gensim.models.Word2Vec(randomorder, sg = 1, size = 100, window=10, iter = 5, seed = i)
    randomwords250=  dict(randommodel.most_similar(positive = [word], topn = 250)).keys()
    randomwords500=  dict(randommodel.most_similar(positive = [word], topn = 500)).keys()
    randomwords1000=  dict(randommodel.most_similar(positive = [word], topn = 1000)).keys()
    
    t2secondmodel = gensim.models.Word2Vec(t2second, sg = 1, size = 100, window=10, iter = 5, seed = i)
    t2secondwords250=  dict(t2secondmodel.most_similar(positive = [word], topn = 250)).keys()
    t2secondwords500=  dict(t2secondmodel.most_similar(positive = [word], topn = 500)).keys()
    t2secondwords1000=  dict(t2secondmodel.most_similar(positive = [word], topn = 1000)).keys()
    
    t1score250 = score(baserandomwords250, t1secondwords250)
    t2score250 = score(baserandomwords250, t2secondwords250)
    randomscore250 = score(baserandomwords250, randomwords250)
    
    t1score500 = score(baserandomwords500, t1secondwords500)
    t2score500 = score(baserandomwords500, t2secondwords500)
    randomscore500 = score(baserandomwords500, randomwords500)

    t1score1000 = score(baserandomwords1000, t1secondwords1000)
    t2score1000 = score(baserandomwords1000, t2secondwords1000)
    randomscore1000 = score(baserandomwords1000, randomwords1000)
    
    t1250.append(t1score250); tr250.append(randomscore250); t2250.append(t2score250)
    t1500.append(t1score500); tr500.append(randomscore500); t2500.append(t2score500)
    t11000.append(t1score1000); tr1000.append(randomscore1000); t21000.append(t2score1000)

    print "250", t1250[-1], tr250[-1], t2250[-1]
    print "500", t1500[-1], tr500[-1], t2500[-1]
    print "1000", t11000[-1], tr1000[-1], t21000[-1]



    #t1score=  len(np.intersect1d(baserandomwords100, t1secondwords))/(1.0*len(np.union1d(baserandomwords100, t1secondwords)))
    #t2score=  len(np.intersect1d(baserandomwords100, t2secondwords))/(1.0*len(np.union1d(baserandomwords100, t2secondwords)))
    #randomscore = len(np.intersect1d(baserandomwords100, randomwords))/(1.0*len(np.union1d(baserandomwords100, randomwords)))
    #t1.append(t1score)
    #tr.append(randomscore)
    #t2.append(t2score)
    
    #print t1[-1], tr[-1], t2[-1]

    
    #cosine(*2otalmodel.wv[word], totalmodel['eye'])
    '''
    t1score, t2score = 0, 0
    for t in t1words:
        t1score += cosine(totalmodel.wv[word], totalmodel[t])
    for t in t2words:
        t2score += cosine(totalmodel.wv[word], totalmodel[t])
    '''



import pandas as pd

new_df = pd.DataFrame([t1250, tr250, t2250, t1500, tr500, t2500, t11000, tr1000, t21000]).T    
new_df.columns = ['t1250', 'tr250', 't2250', 't1500', 'tr500', 't2500', 't11000', 'tr1000', 't21000']

new_df.to_csv('t1second112 vs all 100 iter 250 500 1000.csv')
#t1recent = np.sum(np.array(t1) >= np.array(t2))
#t2recent = np.sum(np.array(t1) < np.array(t2))

# random 100 runs 300d 54 46n
plt.plot(np.arange(0, 50, 1), tr, linestyle = 'dashed')
plt.plot(np.arange(0, 50, 1), t1, linestyle = 'dotted', color = 'green')
plt.plot(np.arange(0, 50, 1), t2, color = 'red')
#plt.savefig('50IterationsUoI.png', dpi = 1200)


#differences_t1_second = np.array(t1) - np.array(t2)
#differences_random = np.array(t1) - np.array(t2)
#differences_t2_second = np.array(t1) - np.array(t2)

import matplotlib.pyplot as plt

def graph_it(tr, t1, t2, ybeg, yend, xlim):
    #trm = np.median(tr)
    #t1m = np.median(t1)
    #t2m = np.median(t2)
    trm = np.mean(tr)
    t1m = np.mean(t1)
    t2m = np.mean(t2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(ybeg, yend)
    ax.plot(np.arange(0, xlim, 1), tr, linestyle = 'dashed', color = 'blue')
    ax.plot(np.arange(0, xlim, 1), np.repeat(trm, xlim), color = 'blue')
    #ax.annotate('random mean', xy = (0, trm), color = 'blue')
    #ax.annotate('random (' + str(round(trm, 4)) + ')', xy = (xlim-1, trm), color = 'blue')
    ax.annotate('random', xy = (xlim, trm), color = 'blue')
    #ax.annotate('(' + str(round(trm, 4)) + ')', xy = (xlim, trm+0.01), color = 'blue')
    ax.plot(np.arange(0, xlim, 1), t1, linestyle = 'dotted', color = 'green')
    ax.plot(np.arange(0, xlim, 1), np.repeat(t1m, xlim), color = 'green')
    #ax.annotate('t1 second (' + str(round(t1m, 4)) + ')', xy = (xlim-1, t1m), color = 'green')
    ax.annotate('t1 second', xy = (xlim, t1m), color = 'green')
    #ax.annotate('(' + str(round(t1m, 4)) + ')', xy = (xlim, t1m+0.005), color = 'green')

    ax.plot(np.arange(0, xlim, 1), t2, color = 'm', linestyle = '-.')
    ax.plot(np.arange(0, xlim, 1), np.repeat(t2m, xlim), color = 'm')
    ax.annotate('t2 second', xy = (xlim, t2m), color = 'm')
    
    #ax.annotate('Random: ' + str(round(trm, 4)), xy = (xlim-98, ybeg + 0.005), color = 'blue')
    
    #ax.annotate('t2 second (' + str(round(t2m, 4)) + ')', xy = (xlim-1, t2m), color = 'm')
    ax.set_title("Random vs. T2 Second(Meganta), Random (Blue), T1 Second (Green) \n Mean Lines", fontdict= {'fontsize': 9 } )
    plt.savefig("100_iterations_random_vs_all_mean_1000neigh.png", dpi = 600)


graph_it(tr1000, t11000, t21000, 0.4, 0.55, 100)

def graph_it2(tr, t1, t2, ybeg, yend, xlim):
    trm = np.median(tr)
    t1m = np.median(t1)
    t2m = np.median(t2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(ybeg, yend)
    ax.plot(np.arange(0, xlim, 1), tr, linestyle = 'dashed', color = 'blue')
    ax.plot(np.arange(0, xlim, 1), np.repeat(trm, xlim), color = 'blue')
    #ax.annotate('random mean', xy = (0, trm), color = 'blue')
    #ax.annotate('random (' + str(round(trm, 4)) + ')', xy = (xlim-1, trm), color = 'blue')
    ax.annotate('random', xy = (xlim, trm), color = 'blue')
    #ax.annotate('(' + str(round(trm, 4)) + ')', xy = (xlim, trm+0.01), color = 'blue')
    ax.plot(np.arange(0, xlim, 1), t1, linestyle = 'dotted', color = 'green')
    ax.plot(np.arange(0, xlim, 1), np.repeat(t1m, xlim), color = 'green')
    #ax.annotate('t1 second (' + str(round(t1m, 4)) + ')', xy = (xlim-1, t1m), color = 'green')
    ax.annotate('t1 second', xy = (xlim, t1m), color = 'green')
    #ax.annotate('(' + str(round(t1m, 4)) + ')', xy = (xlim, t1m+0.005), color = 'green')

    ax.plot(np.arange(0, xlim, 1), t2, color = 'm', linestyle = '-.')
    ax.plot(np.arange(0, xlim, 1), np.repeat(t2m, xlim), color = 'm')
    ax.annotate('t2 second', xy = (xlim, t2m), color = 'm')
    
    #ax.annotate('Random: ' + str(round(trm, 4)), xy = (xlim-98, ybeg + 0.005), color = 'blue')
    
    #ax.annotate('t2 second (' + str(round(t2m, 4)) + ')', xy = (xlim-1, t2m), color = 'm')
    ax.set_title("Topic 2 Second vs. T2 Second(Meganta), Random (Blue), T1 Second (Green)", fontdict= {'fontsize': 9 } )
    plt.savefig("100_iterations_random_vs_all_median_250neigh.png", dpi = 600)











plt.savefig('10_Iterations_t2second_with_median.png', dpi = 800)


#plt.plot(np.arange(0, 10, 1), np.zeros(10), color = 'black', linewidth = 3)

#plt.savefig('50Iterations1.png', dpi = 1200)

'''
plt.figure()
plt.ylim(-10, 10)
plt.plot(np.arange(0, 10, 1), np.repeat(differences_t1_second)
plt.plot(np.arange(0, 10, 1), np.repeat(differences_t2_second, 1000))
plt.plot(np.arange(0, 10, 1), np.repeat(differences_random, 1000))
'''