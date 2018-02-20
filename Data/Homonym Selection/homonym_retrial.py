#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:54:05 2017

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

def retrieve_sentences(topic):
    sent =[]
    for t in topic:
        for te in t:
            if '[S]' in te:
                sent.append(te[4:].split())
    return sent

docs = os.listdir('Tasa Documents/')

topic1= open('Tasa Documents/homeeconomics.txt', 'rb')
topic1 = pickle.load(topic1)

topic2= open('Tasa Documents/miscellaneous.txt', 'rb')
topic2 = pickle.load(topic2)

topic3= open('Tasa Documents/languagearts.txt', 'rb')
topic3 = pickle.load(topic3)

topic4= open('Tasa Documents/business.txt', 'rb')
topic4 = pickle.load(topic4)

topic5= open('Tasa Documents/science.txt', 'rb')
topic5 = pickle.load(topic5)

topic6= open('Tasa Documents/health.txt', 'rb')
topic6 = pickle.load(topic6)

topic7= open('Tasa Documents/socialstudies.txt', 'rb')
topic7 = pickle.load(topic7)

topic8= open('Tasa Documents/industrailarts.txt', 'rb')
topic8 = pickle.load(topic8)

homeeconomics = retrieve_sentences(topic1)
miscellaneous = retrieve_sentences(topic2)
languagearts = retrieve_sentences(topic3)
business = retrieve_sentences(topic4)
science = retrieve_sentences(topic5)
health = retrieve_sentences(topic6)
socialstudies = retrieve_sentences(topic7)
industrialarts = retrieve_sentences(topic8)

def senses(topic, word):
    sense_count = 0
    for t in topic:
        if word in t:
            sense_count+=1
    return sense_count

all_topics = [homeeconomics, miscellaneous, languagearts, business, science, health, socialstudies, industrialarts]
topic_name= ['homeeconomics', 'miscellaneous', 'languagearts', 'business', 'science', 'health', 'socialstudies', 'industrialarts']


word = homos[0]
print word
for a in range(0, len(all_topics)):
    print topic_name[a], senses(all_topics[a], word)
    
def view_sentences(topic, word):
    for t in topic:
        if word in t:
            print t
            

word = homos[26]
print word            

view_sentences(all_topics[0], word)    
view_sentences(all_topics[1], word)
view_sentences(all_topics[2], word)    
view_sentences(all_topics[3], word)
view_sentences(all_topics[4], word)    
view_sentences(all_topics[5], word)
view_sentences(all_topics[6], word)    
view_sentences(all_topics[7], word)
            
        
