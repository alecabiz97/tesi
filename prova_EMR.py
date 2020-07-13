# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:37:59 2020

@author: aleca
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
from ROCCHIO import *
import time
import random
import pickle
import matlab.engine

    
    
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'
#Cross Dataset
DirDukeFromMarket='..\\FeatureCNN\\CrossDataset\\DukeFromMarket_feature.pkl'
DirMarketFromDuke='..\\FeatureCNN\\CrossDataset\\MarketFromDuke_feature.pkl'



#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)
Bayes=loadFile('..\\Bayes_Market_trained.pkl')

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

#Cross dataset
#f=open(DirDukeFromMarket,'rb')
#test_feature, query_feature=pickle.load(f)
#f.close()

gallery,g_id,g_cams=test_feature[0::],test_id[0::],test_cams[0::]
query,q_id,q_cams=query_feature[0:30],query_id[0:30],query_cams[0:30]

print('START')
ranks_index,ranks_sim,ranks_label = calculateRanks_Similarity(query,gallery,g_id,Bayes)
print('Ranks calcolato')

#Calcolo CMC
ranks_label=np.array(g_id)[ranks_index]
cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)

#Calcolo mAP
mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)

print(mAP,cmc_vector[0])

#Seleziono i campioni pos e neg
 #pos,neg=np.zeros_like(ranks_index),np.zeros_like(ranks_index)
k=25
positive,negative=np.zeros((len(q_id),k)),np.zeros((len(q_id),k))
#positive,negative=[],[]
for i in range(len(q_id)):
    pos=[index for index in ranks_index[0:k,i] if g_id[index]==q_id[i]]
    neg=[index for index in ranks_index[0:k,i] if g_id[index]!=q_id[i]]
    positive[i,:]=np.array(pos)+1
    negative[i,:]=np.array(neg)+1
#    positive.append(pos)
#    negative.append(neg)

#Matlab adatto la gallery
#gallery=matlab.double(gallery.tolist())
print('Start matlab')

#positive=np.array([9,19,29,39,59,69,79,89])+1
#negative=np.array([91,191,291,391,591,691,791,891])+1

#start=time.time()
#r_index=np.zeros_like(ranks_index)
#for i in range(query.shape[0]):
#    scores, r_index = eng.EMR(matlab.double(query[i].tolist()),0,matlab.uint16(positive.tolist()), matlab.uint16(negative.tolist()), nargout=2)
#end=time.time()
#print('Tempo ' + str(end-start))
  
eng = matlab.engine.start_matlab()
tempo,ranks=eng.prova(matlab.uint16(positive),matlab.uint16(negative))
print('Tempo' + str(tempo))

ranks_index=np.array(ranks)-1

#Calcolo CMC
ranks_label=np.array(g_id)[ranks_index]
cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)

#Calcolo mAP
mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)

print(mAP,cmc_vector[0])














