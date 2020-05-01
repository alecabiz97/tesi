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
from evaluation import *
from Lbp import *
from hog import *
from BayesianModel import *
import time
import random


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

print('TRAINING COMPLETE')

gallery,gallery_id=test_feature,test_id
query,query_id = query_feature[0:1000:20], query_id[0:1000:20]
print('START TEST')
   
start=time.time()
vettori_cmc,ranks,mAP_list=[],[],[]    
for i in range(2):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,B)
    print('Ranks calcolato')
    ranks.append(ranks_label)
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    
    r1,r5,r10,r20,r50=cmc_vector[[0,4,9,19,49]]
    print('Rank 1: {} - Rank 5: {} - Rank 10: {} - Rank 20: {} - Rank 50: {}  '.format(r1,r5,r10,r20,r50))
    print('mAP: {}'.format(mAP))
    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,5) 
    print('Nuova query calcolata')

end=time.time()
tempo=end-start
print(tempo)
#
#

n,i=200,0
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





    




