# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:45 2020

@author: AleCabiz
"""

import os
import glob
import numpy as np
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
#import ml_metrics as metrics



DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)
Bayes=loadFile('..\\Bayes_Market_trained.pkl')

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

gallery,g_id=test_feature,test_id
query,q_id=query_feature[0:20],query_id[0:20]

r1,r2,r3=calculateRanks_Similarity(query,gallery,g_id,Bayes)
x1=calculateCmcFromRanks(r3,q_id)
m1=calculate_mAP(r3,q_id,len(r3))




#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
##Bayes=loadFile('..\\Bayes_Duke_trained.pkl')
#Bayes.calculateProbBayes(0.19)
#print(min(Bayes.d_differentId))
#print('TRAINING COMPLETE')
##Bayes.plotTrainingHistogram(True)
#d=np.arange(0,3,0.005)
#s=[1/(1+i) for i in d]
#p=[Bayes.calculateProbBayes(i) for i in d]
#X=np.array([d,s,p])
#
#pl.plot(d,s,label='Similarità')
#pl.plot(d,p,label='Probabilità sameId')
#pl.title('Market')
#pl.legend()
#pl.grid()
#pl.show()