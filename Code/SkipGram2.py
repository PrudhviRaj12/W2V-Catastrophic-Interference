#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:50:43 2017

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
    from_h_o = np.dot(from_i_h, weights_h_o)[0]
#    print from_h_o.shape
    pre_left = softmax(from_h_o[0])
    pre_right = softmax(from_h_o[1])
    print (np.max(pre_left), words[np.argmax(pre_left)], input_word, words[np.argmax(pre_right)], np.max(pre_right))

def test_multi(input_word):
    from_i_h = np.dot(input_dic[input_word], weights_i_h)
    from_h_o = np.dot(from_i_h, weights_h_o)[0]
#    print from_h_o.shape
    pre_left = softmax(from_h_o[0])
    left_words = np.argsort(pre_left)
    pre_right = softmax(from_h_o[1])
    right_words = np.argsort(pre_right)
    
    for i in zip(words[left_words[-4 : -1]], words[right_words[-4 : -1]]):
        print (i[0], input_word, i[1])

    #output = softmax(from_h_o)
    #return words[np.argmax(output)], np.max(output)    
    
def get_vectors(words, weights_i_h):
    vect_dict = {}
    for w in words:
        vect_dict[w] = np.dot(input_dic[w], weights_i_h)
    return vect_dict

def cross_entropy(pred, true):
    result = np.sum(true * np.log(pred + 1e-5))
    return -result

context_words = []

for s in selected:
    se = s.split()
    context_words.append([se[1], [se[0], se[2]]])


pca = PCA(n_components= 2, random_state= 2)

input_neurons = output_neurons = vocabulary_size
hidden_neurons = 100

weights_i_h = np.random.rand(input_neurons, hidden_neurons)
weights_h_o = np.random.rand(hidden_neurons, output_neurons)
learning_rate = 0.005

for i in range(0, 1000):
    all_pred = []
    all_out = []
    for j in range(0, len(context_words)):
        current = context_words[j]
        inp = input_dic[current[0]]
        c_left = input_dic[current[1][0]]
        c_right = input_dic[current[1][1]]
        out = np.vstack((c_left, c_right))
        
        #fp 
        
        from_i_h = np.dot(inp, weights_i_h)
        from_h_o = np.dot(from_i_h, weights_h_o)
        pred_output = softmax(from_h_o)
        
        all_pred.append(pred_output)
        all_out.append(out)
        #bp
        
        delta_o = derivative_softmax(pred_output) * np.mean((pred_output - out), axis = 0)
        #delta_o = np.mean(delta_o, axis = 0)
        delta_h = np.dot(weights_h_o, delta_o.T)
        weights_h_o += -learning_rate * from_h_o * delta_o
        weights_i_h += -learning_rate * inp.T * delta_h.T
    
    print i, cross_entropy(np.array(all_pred), np.array(all_out))

#rand = np.random.randint(0, len(context_words))

    
    word_vectors = get_vectors(words, weights_i_h)
    vectors = np.array(word_vectors.values())[:, 0, :]
    trans = pca.fit_transform(vectors)
             
    x = trans[:, 0]         
    y = trans[:, 1]
    names = word_vectors.keys()
             
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title("Iteration: " +str(i))
    for t, text in enumerate(names):
        ax.annotate(text, (x[t], y[t]))
    fig.savefig("Skgram2_100_iter_" + str(i))
    plt.close(fig)
