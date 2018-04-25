#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:42:02 2018

@author: prudhvi
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:20:40 2018

@author: prudhvi
"""


import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from scipy.spatial.distance import cosine

os.chdir('Desktop')

collecter = open('new_corpus.txt', 'r').read().splitlines()

#collecter = [']

appender = []
for c in collecter:
    appender += [c] * 500

half_appender = []
for c in collecter:
    half_appender += [c] * 250

def data_splitter(data):
    vehicles = []
    dinnerware = []
    for c in data:
        if c.split()[1] == 'catch' or c.split()[1] == 'eat':
            vehicles.append(c)
        elif c.split()[1] == 'play' or c.split()[1] == 'pluck':
            dinnerware.append(c)
    return vehicles, dinnerware

fish, instrument = data_splitter(appender)
half_vehicles, half_dinnerware = data_splitter(half_appender)

#np.random.shuffle(vehicles)
#np.random.shuffle(half_dinnerware)
#data = dinnerware + vehicles

data = fish + instrument
np.random.shuffle(data)
#new_data = []
#for d in data:
#    if len(d.split()) == 3:
#        new_data.append(d)
#
#selected = new_data    
all_words = []

for s in data:
    for se in s.split():
        all_words.append(se)

words = np.unique(all_words)
vocabulary_size = len(words)


enc = LabelBinarizer()

ohe = enc.fit_transform(words)

input_dic = {}
for w in range(0, len(words)):
    input_dic[words[w]] = np.reshape(ohe[w], (1, vocabulary_size))
    
context_words = []

for s in data:
    se = s.split()
    context_words.append([se[2], [se[1], se[0]]])
    #context_words.append([se[1], [se[0], se[2]]])
    #context_words.append([se[2], [se[0], se[1]]])

#for s in selected:
#    se = s.split()
#    context_words.append([se[1], [se[0], se[2]]])

input_array = []
output_array = []
for j in range(0, len(context_words)):
#rand = np.random.randint(0, len(context_words))        
    current_exp = context_words[j]
    c_left = input_dic[current_exp[1][0]]
    c_right = input_dic[current_exp[1][1]]
    output = input_dic[current_exp[0]]
    inp = np.vstack((c_left, c_right))
    #inp = (c_left + c_right)/2.
    input_array.append(inp)
    output_array.append(output)

input_feed = np.array(input_array, dtype = np.float32)    
output_feed = np.array(output_array,  dtype = np.float32)


tf.reset_default_graph()

#X = tf.placeholder(shape= [None, vocabulary_size], dtype= tf.float32)
#y = tf.placeholder(shape= [None, vocabulary_size], dtype= tf.float32)

X = tf.placeholder(shape= [2, vocabulary_size], dtype= tf.float32)
y = tf.placeholder(shape= [1, vocabulary_size], dtype= tf.float32)

hidden_dimensions = 300
weights_in = tf.get_variable("weights_in", shape=[vocabulary_size, hidden_dimensions], dtype= tf.float32)
bias_in = tf.get_variable("bias_in", shape=[hidden_dimensions], dtype= tf.float32)
weights_out = tf.get_variable("weights_out",shape=[hidden_dimensions, vocabulary_size], dtype= tf.float32)
bias_out = tf.get_variable("bias_out", shape=[vocabulary_size], dtype= tf.float32)

input_to_hidden = tf.add(tf.matmul(X, weights_in), bias_in)
club_units = tf.reshape(tf.add(input_to_hidden[0, :], input_to_hidden[1, :])/2., (1, hidden_dimensions))
hidden_to_output = tf.matmul(club_units, weights_out) + bias_out
predicted_outputs = tf.nn.softmax(hidden_to_output)

cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicted_outputs)
train_loss = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(0, 10001):
    for i in np.random.randint(0, len(context_words), 100):
        sess.run(optimizer, feed_dict={X:input_feed[i], y:output_feed[i]})
    if e%1000 == 0:
        print (e, sess.run(train_loss, feed_dict={X:input_feed, y:output_feed}))
    
weights_i_h = sess.run(weights_in)
weights_h_o = sess.run(weights_out)

def get_vectors(one_hot_dict, weights):
    save_dict ={}
    for o in one_hot_dict:
        save_dict[o] = np.dot(one_hot_dict[o], weights)
    return save_dict

i2h_vectors = get_vectors(input_dic, weights_i_h)
h2o_vectors = get_vectors(input_dic, weights_h_o.T)

def return_similarity_matrix(word_vectors):
    similarity_dict = defaultdict(dict)
    for w in word_vectors:
        for ww in word_vectors:
            similarity_dict[w][ww] = cosine_similarity(word_vectors[w], word_vectors[ww])[0][0]
            #similarity_dict[w][ww] = cosine(word_vectors[w], word_vectors[ww])

    return similarity_dict

i2h_sim_dict = return_similarity_matrix(i2h_vectors)
h2o_sim_dict = return_similarity_matrix(h2o_vectors)

i2h_dframe = pd.DataFrame(i2h_sim_dict)
h2o_dframe = pd.DataFrame(h2o_sim_dict)

i2h_dframe.to_csv('New Tests/input2hidden_newlang_twosenses_random_run2.csv')
h2o_dframe.to_csv('New Tests/hidden2output_newlang_twosenses_random_run2.csv')

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=123)

vec_vals = tsne.fit_transform(np.array(i2h_vectors.values())[:, 0, :])

import matplotlib.pyplot as plt

plt.scatter(vec_vals[:, 0], vec_vals[:, 1])

   #random smash 0.5186 stop 0.500    
# DV smash 0.497 stop 0.524 | smash 0.011 stop 0.07
#CD 0.5173 stop 0.5362 | smash 0.1033 stop 0.1394