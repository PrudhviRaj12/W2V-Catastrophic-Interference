#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:34:27 2017

@author: prudhvi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random = pd.read_csv("firm_cosinet1second_vs_all_200_runs_50_neigh.csv")

current = random.T[0:4]

def graph_it(tr, t1, t2, ybeg, yend, xlim, filename):
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
    ax.set_title("T1 Second vs. T2 Second(Meganta), Random (Blue), T1 Second (Green) \n k = 1000", fontdict= {'fontsize': 9 } )
    plt.savefig(filename, dpi = 600)


t1 = random['t1s'].values
tr = random['rand'].values
t2 = random['t2s'].values

graph_it(tr, t1, t2, 4, 8, 200, "a.png")