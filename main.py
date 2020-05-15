# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:45 2020

@author: AleCabiz
"""

import os
import glob
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importData import *
from histogram import *
from evaluation import *
from Lbp import *
from hog import *
from BayesianModel import *
from analisi_risultati import *
from queryExpansion import *
import time
import random
import pickle



print('START')

    
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

    
#Load BayesianModel gia addestrato
B=loadFile('..\\B_Market_trained.pkl')
#B=loadFile('..\\B_Duke_trained.pkl')


gallery,gallery_id=test_feature,test_id
query,query_id = query_feature[0:2], query_id[0:2]


    
start=time.time()


print('START TEST')

n,k=3,5
results=[]
vettori_cmc,ranks,mAP_list=[],[],[] 
for i in range(n+1):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,B)
    ranks.append(ranks_label)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    
    
    #Calcolo la nuova query    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,k)
    print('Nuova query calcolata')
k_n_ranks_cmc_mAP=[k,n,ranks,vettori_cmc,mAP_list]
results.append(k_n_ranks_cmc_mAP)
print('####################à')
     
    
end=time.time()
tempo=end-start
print(tempo)





    




