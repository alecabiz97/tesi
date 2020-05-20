# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:13:17 2020

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
import time
import random
import pickle

DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirDuke)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

    
#Load BayesianModel gia addestrato
#B=loadFile('..\\B_Market_trained.pkl')
B=loadFile('..\\B_Duke_trained.pkl')


gallery,gallery_id=test_feature,test_id
query,query_id = query_feature[0:20], query_id[0:20]
 
start=time.time()

ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,B)
end=time.time()
tempo1=end-start
print('v1: {}'.format(tempo1))

start=time.time()
ranks_index2,ranks_probability2,ranks_label2 =calculateRanks2(query,gallery,gallery_id,B)
end=time.time()
tempo2=end-start
print('v2: {}'.format(tempo2))






ds=B.d_sameId
dd=B.d_differentId
print(len(ds))
s=0
for i in tr:
    cnt=train_id.count(i)
    s += (cnt*(cnt-1))/2

print(s)


