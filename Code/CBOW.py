#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:21:46 2017

@author: prudhvi
"""

import numpy as np
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

os.chdir("PyT")

data = open('Elman_Corpus.txt').read().splitlines()
np.random.seed(2)
# remove sentences with less than 3 words

def softmax(x):
    exps = np.exp(x - x.max())
    return exps/(np.sum(exps))

def derivative_softmax(x):
    return (x) * (1 - x)


new_data = []
for d in data:
    if len(d.split()) == 3:
        new_data.append(d)

selected = new_data    
all_words = []

for s in selected:
    for se in s.split():
        all_words.append(se)

words = np.unique(all_words)
vocabulary_size = len(words)

input_neurons = output_neurons = vocabulary_size
hidden_neurons = 100

enc = LabelBinarizer()

ohe = enc.fit_transform(words)

input_dic = {}
for w in range(0, len(words)):
    input_dic[words[w]] = np.reshape(ohe[w], (1, vocabulary_size))
    
'''
word_pairs = []

for s in selected:
    split = s.split()
    for se in range(0, len(split)-1):
        print split[se], split[se+1]
        word_pairs.append([split[se], split[se+1]])
        word_pairs.append([split[se+1], split[se]])
'''
def test(input_word):
    from_i_h = np.dot(input_dic[input_word], weights_i_h)
    from_h_o = np.dot(from_i_h, weights_h_o)
    output = softmax(from_h_o)
    return words[np.argmax(output)], np.max(output)    
    
def get_vectors(words, weights_i_h):
    vect_dict = {}
    for w in words:
        vect_dict[w] = np.dot(input_dic[w], weights_i_h)
    return vect_dict

pca = PCA(n_components= 2, random_state= 123)
        
    
weights_i_h = np.random.rand(input_neurons, hidden_neurons)
weights_h_o = np.random.rand(hidden_neurons, output_neurons)
learning_rate = 0.0008
# 1st forward propagation

def cross_entropy(pred, true):
    result = np.sum(true * np.log(pred + 1e-5))
    return -result

context_words = []

for s in selected:
    se = s.split()
    context_words.append([se[1], [se[0], se[2]]])

cross_ent = {}
for i in range(0, 2000):
    all_pred = []
    all_ops = []

    for j in range(0, len(context_words)):
        rand = np.random.randint(0, len(context_words))        
        current_exp = context_words[rand]
        c_left = input_dic[current_exp[1][0]]
        c_right = input_dic[current_exp[1][1]]
        output = input_dic[current_exp[0]]
        
        inp = np.vstack((c_left, c_right))
        
        from_i_h = np.mean(np.dot(inp, weights_i_h), axis =0)
        from_h_o = np.dot(from_i_h, weights_h_o)
        pred_output = softmax(from_h_o)
        
        if i%10 == 0:
            all_pred.append(pred_output)
            all_ops.append(output)
            
        delta_o = derivative_softmax(pred_output) * (pred_output - output)
        delta_h = np.dot(weights_h_o, delta_o.T)
        weights_h_o += -learning_rate * from_h_o * delta_o
        weights_i_h += -learning_rate * np.reshape(np.mean(inp, axis = 0), (1, 19)).T * delta_h.T
    #print(np.max(pred_output)), words[np.argmax(pred_output)]

    if i%10 == 0:
        all_pred = np.reshape(all_pred, (len(context_words), vocabulary_size))
        all_ops = np.reshape(all_ops, (len(context_words), vocabulary_size))
        cros_en = cross_entropy(all_pred, all_ops)
        cross_ent[i+8000] = cros_en
        print i, cros_en
            
        
        word_vectors = get_vectors(words, weights_i_h)
    
        vectors = np.array(word_vectors.values())[:, 0, :]
        trans = pca.fit_transform(vectors)
    
        x = trans[:, 0]
        y = trans[:, 1]
        names = word_vectors.keys()
    
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title("Iteration: " +str(i+8000))
    
        for t, text in enumerate(names):
            ax.annotate(text, (x[t], y[t]))
        fig.savefig("cbow_100_iter_" +str(i+8000))
        plt.close(fig)


import pickle
filename = open('CBOW_100d_10000iter_weights.txt', 'wb')
pickle.dump(weights_i_h, filename)