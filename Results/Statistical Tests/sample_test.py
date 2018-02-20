# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 21:21:50 2018

@author: prudh
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import os
os.chdir('C:\Users\prudh\Desktop\New Tabs')



word = 'slip'
topic1 = 'science'
topic2 = 'business'
overlap = 'overlap'
#overlap = 'no_overlap'

from collections import defaultdict

wordlist = pd.read_csv('wordlist.csv')

dicto = defaultdict(list)

for w in range(0, len(wordlist)):
    a = wordlist['word'][w]
    b = wordlist['topic 1'][w]
    c = wordlist['topic 2'][w]
    
    dicto[a] = [b, c]

testframe = pd.DataFrame()
for w in dicto.keys():
    if w == 'compact ':
        word = 'compact'
    else:
        word  = w
    print word
    topic1 = dicto[w][0]
    topic2 = dicto[w][1]
    overlap = 'overlap'
    
    #print w, topic1, topic2

    filename = 'base_' + topic1 + '_' + topic2 +'_word_' + word + '_' + overlap + '.csv'
    print filename
    dframe = pd.read_csv(filename)

    for k in [100, 200, 300]:
        #random = 'random_' + str(k) 
        topic1_second = topic2 + '_' + topic1 + '_' + str(k)
        topic2_second = topic1 + '_' + topic2 + '_' + str(k)
        
        base = dframe[topic2_second].values
        
        #comp1 = dframe[topic1_second].values
        comp = dframe[topic1_second].values
        
        t, p = ttest_rel(base, comp)
        
        #print 'random', 'topic 1 second', k
        if p < 0.05:
             testframe = testframe.append(pd.DataFrame([word, 'topic 2 second', 'topic 1 second', str(k), 'significant']).T)
    #        print 'significant'
        else:
             testframe = testframe.append(pd.DataFrame([word, 'topic 2 second', 'topic 1 second', str(k), 'not significant']).T)
            #print 'not significant'
            
#        t, p = ttest_rel(base, comp2)
#        if p < 0.05:
#            testframe =testframe.append(pd.DataFrame([word, 'random', 'topic 2 second', str(k), 'significant']).T)
#    #        print 'significant'
#        else:
#            testframe = testframe.append(pd.DataFrame([word, 'random', 'topic 2 second', str(k), 'not significant']).T)
#            #print 'not significant'
testframe.columns = ['word', 'base', 'comparison', 'k', 'significance']
testframe.to_csv('ttest_topic2_second_with_no_overlap.csv')
            
    