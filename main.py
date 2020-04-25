# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:45 2020

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



print('START')

start=time.time()
    
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Load Duke
testDuke,queryDuke,trainDuke=loadCNN(DirDuke)
gallery_cams_Duke, gallery_cnn_Duke, gallery_id_Duke, gallery_desc_Duke = testDuke
query_cams_Duke, query_cnn_Duke, query_id_Duke, query_desc_Duke = queryDuke
train_cams_Duke, train_cnn_Duke, train_id_Duke, train_desc_Duke = trainDuke

#Load Market
testMarket,queryMarket,trainMarket=loadCNN(DirMarket)
gallery_cams_Market, gallery_cnn_Market, gallery_id_Market, gallery_desc_Market = testMarket
query_cams_Market, query_cnn_Market, query_id_Market, query_desc_Market = queryMarket
train_cams_Market, train_cnn_Market, train_id_Market, train_desc_Market = trainMarket

print('Feature importate')

train,train_id=train_cnn_Market,train_id_Market
gallery,gallery_id=gallery_cnn_Market[0:600],gallery_id_Market[0:600]
query,query_id = query_cnn_Market[0:20], query_id_Market[0:20]


#print('START Duke')
#B_Duke=BayesianModel()
#B_Duke.train(train_cnn_Duke[0:1000],train_id_Duke[0:1000])
#

#print('START Market')
#B_Market=BayesianModel()
#B_Market.train(train_cnn_Market,train_id_Market)

print('TRAINING COMPLETE')

h1,b1=B_Market.hist_d_sameId.copy()
h2,b2=B_Market.hist_d_differentId.copy()

x1=np.zeros_like(h1,float)
x2=np.zeros_like(h2,float)
for i in range(len(b1)-1):
    x1[i]=(b1[i] + b1[i+1])/2
    x2[i]=(b2[i] + b2[i+1])/2

width_binsSame=(max(b1)-min(b1))/100
width_binsDiff=(max(b2)-min(b2))/100

h1_n=(h1/np.sum(h1))/width_binsSame
h2_n=(h2/np.sum(h2))/width_binsDiff

#h1_n=h1/np.sum(h1)
#h2_n=h2/np.sum(h2)

pl.bar(x1,h1_n,width_binsSame,label='sameId',color='r')
pl.bar(x2,h2_n,width_binsDiff,label='differentId',color='b')


pl.legend()
pl.xlabel('Distance')
pl.show()


print('START TEST')

    
vettori_cmc=[]    
for i in range(3):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,B_Market)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    
    vettori_cmc.append(cmc_vector)
    #plot_CMC(cmc_vector)
    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,5) 
    print('Nuova query calcolata')
    
end=time.time()
tempo=end-start
print(tempo)


n,i=10,0
for y in vettori_cmc:
    x=np.arange(len(y[0:n]))+1
    pl.plot(x,y[0:n],label='iterazione {}'.format(i))
    i += 1
pl.title('Cumulative Match Characteristic')
pl.ylabel('Probability of Identification')
pl.xlabel('Rank')
pl.grid(True)
pl.legend()
pl.show()  





    




