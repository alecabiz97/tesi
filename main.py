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
testDuke,queryDuke,trainDuke=loadCNN(DirMarket)
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
gallery,gallery_id=gallery_cnn_Market,gallery_id_Market
query,query_id = query_cnn_Market[0:100], query_id_Market[0:100]


print('START')
#B=BayesianModel()
#B.train(train,train_id)

print('TRAINING COMPLETE')
print('START TEST')


#Calcolo la somma del numero di immagini per ciscuna query nella gallery, la usa per la cmc.
number_istance=0
for y in query_id:
    number_istance += gallery_id.count(y)
    
    
for i in range(3):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,B)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    val_cmc=np.sum(ranks_label==query_id,1).cumsum()
    cmc_vector=val_cmc/number_istance
    
    plot_CMC(cmc_vector)
    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,5) 
    print('Nuova query calcolata')
    
    

end=time.time()
tempo=end-start
print(tempo)



 

   


    




