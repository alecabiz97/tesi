# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:45 2020

@author: AleCabiz
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from utils.Lbp import *
from utils.hog import *
from utils.BayesianModel import *
from analisi_risultati import *
from utils.queryExpansion import *
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

gallery,g_id,g_cams=test_feature,test_id,test_cams
query,q_id,q_cams=query_feature[0:50],query_id[0:50],query_cams[0:50]

print('START')

results=[]
resultsRanks=[]
n,k=2,5
for i in range(n):
    ranks_index,ranks_sim,ranks_label = calculateRanks_Similarity(query,gallery,g_id,Bayes)
    print('Ranks calcolato')
    
    #Calcolo CMC
    ranks_label=np.array(g_id)[ranks_index]
    cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)
    
    #Calcolo mAP
    mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
    
    print((mAP,cmc_vector[0]))
    query=queryExpansion_CameraFilter(ranks_index,ranks_sim,gallery,g_cams,query,q_cams,k,AQE=False,soglia=0.5)
#    query=queryExpansion(ranks_index,ranks_sim,gallery,query,k,AQE=False,soglia=0.5)
    #Feedback
#    query=queryExpansion_withFeedback(ranks_index,ranks_sim,ranks_label,gallery,query,q_id,k,probEquals1=True)
#    query=queryExpansion_withFeedback_CameraFilter(ranks_index,ranks_sim,ranks_label,gallery,g_cams,query,query_id,q_cams,k,probEquals1=True)

    
#    results.append([k,n,vettori_cmc,vettore_mAP])
#    resultsRanks.append([k,n,Ranks_index])
#
#risultatiTest=[len(set(q_id)),q_id,results]
#risultatiRanks=[len(set(q_id)),q_id,resultsRanks]        


