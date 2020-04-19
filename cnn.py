# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:14:33 2020

@author: AleCabiz
"""

import os
import glob
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importData import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
from BayesianModel import *
import time
import random

DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#CnnMarket=loadCNN(DirMarket)
#CnnDuke=loadCNN(DirDuke)

#Duke
g_cams,g_feature,g_id=CnnMarket[0:3]
q_cams,q_feature,q_id=CnnMarket[4:7]


indeces=np.arange(0,40000,1)
n=5000
y=[g_id[i] for i in indeces[0:n]]
x=[g_feature[i] for i in indeces[0:n]]


B=BayesianModel()
B.train(x,y)
print('TRAINING COMPLETE')

h1,b1=B.hist_d_sameId
h2,b2=B.hist_d_differentId 

x1=np.zeros_like(h1)
x2=np.zeros_like(h2)
for i in range(len(b1)-1):
    x1[i]=(b1[i] + b1[i+1])/2
    x2[i]=(b2[i] + b2[i+1])/2
    
width_binsSame=(max(b1)-min(b1))/100
width_binsDiff=(max(b2)-min(b2))/100

#pl.bar(x1,h1,width_binsSame,label='sameId',color='r')
#pl.bar(x2,h2,width_binsDiff,label='differentId',color='b')
pl.plot(x1,h1,label='sameId',color='r')
pl.plot(x2,h2,label='differentId',color='b')
 
pl.legend()
pl.xlabel('Distance')
pl.ylabel('Probability')
pl.show()
