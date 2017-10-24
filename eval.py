# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:08:34 2017

@author: prudh
"""

import numpy as np
import os
import pickle
os.chdir("C:\Users\prudh\W2V-Catastrophic-Interference\GensimImplementation\Vecs")

#No Ordering 2 Senses 1 Iter

vectors = os.listdir("Vehicles_Dinnerware_News 3 Senses 3 Iter")

def load_file(filename):
    files = open("Vehicles_Dinnerware_News 3 Senses 3 Iter/" + filename)
    v1 = pickle.load(files)
    files.close()
    return v1


from scipy.spatial.distance import cosine, correlation
first = load_file(vectors[0])
second = load_file(vectors[1])

query_word=  'break'
check_word = ['car', 'truck', 'glass', 'plate', 'news', 'story']
#check_word = ['car', 'truck', 'glass', 'plate']

from collections import defaultdict

dic = defaultdict(dict)
for v in range(0, 6000):
    first = load_file(vectors[v])
    for c in check_word:
        dic[int(vectors[v].strip('vehi_dinner_news_vectors_.pkl'))][c] = cosine(first[query_word], first[c])
    
import pandas as pd

dframe = pd.DataFrame(dic).T

dframe['Vehicles'] = (dframe['car'] + dframe['truck'])/2
dframe['Dinnerware'] = (dframe['glass'] + dframe['plate'])/2
dframe['News'] = (dframe['news'] + dframe['story'])/2

dframe['closer to vehicles'] = (dframe['Vehicles'] < dframe['Dinnerware']) & (dframe['Vehicles'] < dframe['News']) 
dframe['closer to dinnerware'] = (dframe['Dinnerware'] < dframe['Vehicles']) & (dframe['Dinnerware'] < dframe['News']) 
dframe['closer to news'] = (dframe['News'] < dframe['Dinnerware']) & (dframe['News'] < dframe['Vehicles']) 

print "Vehicle Occuring First: " + str(sum(dframe['closer to vehicles']))
print "Dinnerware Occuring First: " + str(sum(dframe['closer to dinnerware']))
print "News Occuring First: " + str(sum(dframe['closer to news']))

print "Proportion of Vehicle Occuring First: " + str(sum(dframe['closer to vehicles'])/(len(vectors) + 0.0))
print "Proportion of Dinnerware Occuring First: " + str(sum(dframe['closer to dinnerware'])/(len(vectors) + 0.0))
print "Proportion of News Occuring First: " + str(sum(dframe['closer to news'])/(len(vectors) + 0.0))


dframe.to_csv("No Ordering 3 Senses 3 Iter.csv")
