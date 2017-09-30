#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:49:10 2017

@author: prudhvi
"""
import numpy as np
import os
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
hidden_neurons = 20

from sklearn.preprocessing import LabelBinarizer

enc = LabelBinarizer()

ohe = enc.fit_transform(words)

input_dic = {}
for w in range(0, len(words)):
    input_dic[words[w]] = np.reshape(ohe[w], (1, vocabulary_size))
    
word_pairs = []

for s in selected:
    split = s.split()
    for se in range(0, len(split)-1):
        print split[se], split[se+1]
        word_pairs.append([split[se], split[se+1]])
        word_pairs.append([split[se+1], split[se]])

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
    

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components= 2, random_state= 123)
        
    
weights_i_h = np.random.rand(input_neurons, hidden_neurons)
weights_h_o = np.random.rand(hidden_neurons, output_neurons)
learning_rate = 0.0005
# 1st forward propagation

def cross_entropy(pred, true):
    result = np.sum(true * np.log(pred + 1e-5))
    return -result

cross_ent = {}
for i in range(1, 20001):
    
    all_pred = []
    all_ops = []
    
    for j in range(0, len(word_pairs)):
        rand = np.random.randint(0, len(word_pairs))
    
        samp_inp = input_dic[word_pairs[rand][0]]
        samp_out = input_dic[word_pairs[rand][1]]

    # forward    
        from_i_h = np.dot(samp_inp, weights_i_h)
        from_h_o = np.dot(from_i_h, weights_h_o)
        pred_output = softmax(from_h_o)
        
        if i%100 == 0:
            all_pred.append(pred_output)
            all_ops.append(samp_out)
        #backprop    
        delta_o = derivative_softmax(pred_output) * (pred_output - samp_out)
        delta_h = np.dot(weights_h_o, delta_o.T)
        weights_h_o += -learning_rate * from_h_o * delta_o

    #print delta_h.shape, delta_o.shape
        weights_i_h += -learning_rate * samp_inp.T * delta_h.T
    
    if i%100 == 0:
        
        all_pred = np.reshape(all_pred, (len(word_pairs), vocabulary_size))
        all_ops = np.reshape(all_ops, (len(word_pairs), vocabulary_size))
        cros_en = cross_entropy(all_pred, all_ops)
        cross_ent[i] = cros_en
        print i, cros_en
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
        fig.savefig('all_20_samples_iter_' +str(i) + '.png')
        plt.close(fig)
        

        


word_vectors = get_vectors(words, weights_i_h)

vectors = np.array(word_vectors.values())[:, 0, :]
trans = pca.fit_transform(vectors)


#for t in range(0, 15):
#    print word_pairs[t], test(word_pairs[t][0])


x = trans[:, 0]/np.linalg.norm(trans[:, 0])
y = trans[:, 1]/np.linalg.norm(trans[:, 1])
names = word_vectors.keys()

fig, ax = plt.subplots()
ax.scatter(x, y)

import matplotlib.pyplot as plt

for i, text in enumerate(names):
    ax.annotate(text, (x[i], y[i]))
    
import pickle

filename = open('weights_20000_randomized.txt', 'wb')
pickle.dump(weights_i_h, filename)

import operator

def top_five(word):
    word_to_vec = dict()
    for w in word_vectors:
        if w != word:
            word_to_vec[w] = cosine(word_vectors[word], word_vectors[w])
    sorted_dict = sorted(word_to_vec.items(), key = operator.itemgetter(1))
    return sorted_dict
    
so = top_five('break')
print so    #sort
    
















        
#print words[np.argmax(pred_output)], np.max(pred_output)
# 2nd forward propagation


from_i_h = np.dot(samp_inp.T, weights_i_h)
from_h_o = np.dot(from_i_h, weights_h_o)

pred_output = softmax(from_h_o)

# 2nd back propagation


delta_o = derivative_softmax(pred_output[0]) * (pred_output[0] - samp_out[0])
delta_h = from_i_h * np.dot(weights_h_o, delta_o)

weights_i_h += -learning_rate * samp_inp * delta_h
weights_h_o += -learning_rate * from_h_o * delta_o








