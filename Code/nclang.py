import numpy as np
import pickle
import os

os.chdir('Desktop/w2v')
homolist = open('homonyms_list.txt', 'rb')
homolist = pickle.load(homolist)

homos = homolist.keys()

topic1= open('Tasa Documents/languagearts.txt', 'rb')
topic1 = pickle.load(topic1)

topic2= open('Tasa Documents/health.txt', 'rb')
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

w1 = 'gum'

def score(base, current):
    num = 1.0 * len(np.intersect1d(base, current))
    den = len(np.union1d(base, current))
    return num/den

import gensim

np.random.shuffle(t1_sent)
data = t1_sent + t2_sent
np.random.shuffle(data)
basemodel = gensim.models.Word2Vec(data, sg = 1, size = 300, window = 5, iter = 5, seed = 122)

basewords = basemodel.most_similar(positive = [w1], topn = 300)

basew100 = dict(basewords[0:100]).keys()
basew200 = dict(basewords[0:200]).keys()
basew300 = dict(basewords[0:300]).keys()

basen100 = dict(basewords[0:100]).keys()
basen200 = dict(basewords[100:200]).keys()
basen300 = dict(basewords[200:300]).keys()

t1_second = t2_sent + t1_sent
t2_second = t1_sent + t2_sent
random = t1_sent + t2_sent
np.random.shuffle(random)

rand100, t1s100, t2s100 = [], [], []
rand200, t1s200, t2s200 = [], [], []
rand300, t1s300, t2s300 = [], [], []

randno100, t1no100, t2no100 = [], [], []
randno200, t1no200, t2no200 = [], [], []
randno300, t1no300, t2no300 = [], [], []


for i in range(0, 50):

    print i
    np.random.shuffle(random)    
    randmodel = gensim.models.Word2Vec(random, sg=1, size = 300, window = 5, iter = 5, seed=  i)
    randwords = randmodel.most_similar(positive = [w1], topn = 300)
    
    randw100 = dict(randwords[0:100]).keys()
    randw200 = dict(randwords[0:200]).keys()
    randw300 = dict(randwords[0:300]).keys()

    rand100.append(score(basew100, randw100))
    rand200.append(score(basew200, randw200))
    rand300.append(score(basew300, randw300))
    
    randn100 = dict(randwords[0:100]).keys()
    randn200 = dict(randwords[100:200]).keys()
    randn300 = dict(randwords[200:300]).keys()

    randno100.append(score(basen100, randn100))
    randno200.append(score(basen200, randn200))
    randno300.append(score(basen300, randn300))


    
    t1secondmodel = gensim.models.Word2Vec(t1_second, sg=1, size = 300, window = 5, iter = 5, seed=  i)
    t1secondwords = t1secondmodel.most_similar(positive = [w1], topn = 300)

    t1sw100 = dict(t1secondwords[0:100]).keys()
    t1sw200 = dict(t1secondwords[0:200]).keys()
    t1sw300 = dict(t1secondwords[0:300]).keys()

    t1s100.append(score(basew100, t1sw100))
    t1s200.append(score(basew200, t1sw200))
    t1s300.append(score(basew300, t1sw300))
    
    t1sn100 = dict(t1secondwords[0:100]).keys()
    t1sn200 = dict(t1secondwords[100:200]).keys()
    t1sn300 = dict(t1secondwords[200:300]).keys()
    
    t1no100.append(score(basen100, t1sn100))
    t1no200.append(score(basen200, t1sn200))
    t1no300.append(score(basen300, t1sn300))


   
    t2secondmodel = gensim.models.Word2Vec(t2_second, sg=1, size = 300, window = 5, iter = 5, seed=  i)
    t2secondwords = t2secondmodel.most_similar(positive = [w1], topn = 300)

    t2sw100 = dict(t2secondwords[0:100]).keys()
    t2sw200 = dict(t2secondwords[0:200]).keys()
    t2sw300 = dict(t2secondwords[0:300]).keys()

    t2s100.append(score(basew100, t2sw100))
    t2s200.append(score(basew200, t2sw200))
    t2s300.append(score(basew300, t2sw300))
    
    t2sn100 = dict(t2secondwords[0:100]).keys()
    t2sn200 = dict(t2secondwords[100:200]).keys()
    t2sn300 = dict(t2secondwords[200:300]).keys()
    
    t2no100.append(score(basen100, t2sn100))
    t2no200.append(score(basen200, t2sn200))
    t2no300.append(score(basen300, t2sn300))

    
    if i%10 == 0:
        
        print "OVERLAP"
        print '100'
        print np.mean(rand100), np.mean(t1s100), np.mean(t2s100)
        print '200'
        print np.mean(rand200), np.mean(t1s200), np.mean(t2s200)
        print '300'
        print np.mean(rand300), np.mean(t1s300), np.mean(t2s300)
        
        print "NO OVERLAP"
        print '0-100'
        print np.mean(randno100), np.mean(t1no100), np.mean(t2no100)
        print '100-200'
        print np.mean(randno200), np.mean(t1no200), np.mean(t2no200)
        print '200-300'
        print np.mean(randno300), np.mean(t1no300), np.mean(t2no300)
    

# T2 second running now
import pandas as pd

df = pd.DataFrame([rand100, t1s100, t2s100, rand200, t1s200, t2s200, rand300, t1s300, t2s300]).T
df.columns = ['random_100', 'health_languagearts_100', 'langaugearts_health_100', 
'random_200', 'health_languagearts_200', 'langaugearts_health_200',
'random_300', 'health_languagearts_300', 'langaugearts_health_300']

df.to_csv('New Tabs/base_langaugearts_health_word_gum_overlap.csv')

df = pd.DataFrame([randno100, t1no100, t2no100, randno200, t1no200, t2no200, randno300, t1no300, t2no300]).T
df.columns = ['random_100', 'health_languagearts_100', 'langaugearts_health_100', 
'random_200', 'health_languagearts_200', 'langaugearts_health_200',
'random_300', 'health_languagearts_300', 'langaugearts_health_300']

df.to_csv('New Tabs/base_langaugearts_health_word_gum_no_overlap.csv')

#df.to_csv('New Tabs/base_random_word_plane_no_overlap.csv')


#df.to_csv('New Tabs/mount_industrialarts_science.csv')
