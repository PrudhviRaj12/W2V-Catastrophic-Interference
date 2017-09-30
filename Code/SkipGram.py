#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:27:06 2017

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
weights_h_o = np.random.rand(2, hidden_neurons, output_neurons)
learning_rate = 0.005

cross_ent = {}
for i in range(0, 1000):
    
    all_pred = []
    all_out = []
    for j in range(0, len(context_words)):
        rand = np.random.randint(0, len(context_words))
        current = context_words[rand]
        inp = input_dic[current[0]]
        c_left = input_dic[current[1][0]]
        c_right = input_dic[current[1][1]]
        out = np.vstack((c_left, c_right))
        
        from_i_h = np.dot(inp, weights_i_h)
        from_h_o = np.dot(from_i_h, weights_h_o)[0]
        pre_left = softmax(from_h_o[0])
        pre_right = softmax(from_h_o[1])
        
        #print current, words[np.argmax(pre_left)], np.max(pre_left)
        #print current, words[np.argmax(pre_right)], np.max(pre_right)
        pred_out = np.vstack((pre_left, pre_right))
    
        if i%10 == 0:        
            all_pred.append(pred_out)
            all_out.append(out)
            
        der_left = derivative_softmax(pre_left)
        der_right = derivative_softmax(pre_right)
        der = np.vstack((der_left, der_right))
        
        delta_o = der * (pred_out - out)
        #delta_h = np.dot(weights_h_o, delta_o.T)
        
        c = from_h_o * delta_o
        d = np.reshape(c, (2, 1, vocabulary_size))
        weights_h_o += -learning_rate * d
        
        dh_left = np.dot(weights_h_o[0], delta_o.T[:, 0])
        dh_right = np.dot(weights_h_o[1], delta_o.T[:, 1])
        delta_h = np.stack((dh_left, dh_right), axis = 1)
        delta_h = np.mean(delta_h, axis = 1)
        
        weights_i_h += -learning_rate * inp.T * delta_h

    if i%10 == 0:
        
        print test(context_words[10][0])
        
        all_pred = np.array(all_pred)
        all_out = np.array(all_out)
        cross_ent[i] = cross_entropy(all_pred, all_out)
        print i, cross_ent[i]
        
        '''
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
        fig.savefig("Ski_200_iter_" + str(i))
        plt.close(fig)
        '''


sumer = weights_i_h + np.mean(weights_h_o.T, axis = 2)
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
fig.savefig("Ski_200_iter_" + str(i))
plt.close(fig)





import pickle

filename = open('weights_skipgram_200.txt', "wb")
pickle.dump(weights_i_h, filename)    
plt.scatter(cross_ent.keys(), cross_ent.values())


from scipy.spatial.distance import cosine

current = context_words[0]
inp = current[0]
c_left = current[1][0]
c_right = current[1][1]
  
d1, d2 = 'monster break plate', 'monster smash glass'
s1, s2 = 0, 0

for d in d1.split():
    s1+= word_vectors[d]
for d in d2.split():
    s2 += word_vectors[d]


w2c1 = cosine(word_vectors[inp], word_vectors[c_left])
w2c2 = cosine(word_vectors[inp], word_vectors[c_right])
w2c3 = cosine(word_vectors[c_left], word_vectors[c_right])
#==============================================================================
#         word_vectors = get_vectors(words, weights_i_h)
#         vectors = np.array(word_vectors.values())[:, 0, :]
#         trans = pca.fit_transform(vectors)
#         
#         x = trans[:, 0]
#         y = trans[:, 1]
#         names = word_vectors.keys()
#         
#         fig, ax = plt.subplots()
#         ax.scatter(x, y)
#         #ax.set_xlim(-3, 0)
#         #ax.set_ylim(-2, 1)
#         ax.set_title("Iteration: " +str(i))
#         for t, text in enumerate(names):
#             ax.annotate(text, (x[t], y[t]))
#         fig.savefig('SkipGram_30_iter_spec_' + str(i))
#         plt.close(fig)
# 
#==============================================================================



plt.scatter(cross_ent.keys(), cross_ent.values())


word_vectors = get_vectors(words, weights_i_h)
vectors = np.array(word_vectors.values())[:, 0, :]
trans = pca.fit_transform(vectors)
         
x = trans[:, 0]         
y = trans[:, 1]
names = word_vectors.keys()
         
fig, ax = plt.subplots()
ax.scatter(x, y)
         #ax.set_xlim(-3, 0)
         #ax.set_ylim(-2, 1)
ax.set_title("Iteration: " +str(i))
for t, text in enumerate(names):
    ax.annotate(text, (x[t], y[t]))


#weights_i_h += -learning_rate 

'''
for i in range(0, len(context_words)):
    current = context_words[i]
    inp = input_dic[current[0]]
    c_left = input_dic[current[1][0]]
    c_right = input_dic[current[1][1]]
    out = np.vstack((c_left, c_right))
    
    #fp
    
    from_i_h = np.dot(inp, weights_i_h)
    from_i_h = np.vstack((from_i_h, from_i_h))
    from_h_o = np.dot(from_i_h, weights_h_o)
    pred_out = np.vstack((softmax(from_h_o[0]), softmax(from_h_o[1])))
    vals = np.argmax(pred_out, axis=1)
    nums = np.max(pred_out, axis = 1)
    print words[vals[0]], words[vals[1]], nums
               
    #bp
    
    delta_o = derivative_softmax(pred_out) * (pred_out - out)
    #print delta_o
    delta_h = np.dot(weights_h_o, delta_o.T)
    #delta_o = np.reshape(np.mean(delta_o, axis = 1), (vocabulary_size, 1))
    weights_h_o += -learning_rate * np.sum(from_h_o *  delta_o, axis = 0)
    
    d = np.sum(delta_h, axis = 1)
    d = np.reshape(d, (50, 1))
    weights_i_h += -learning_rate * inp.T * d.T
'''