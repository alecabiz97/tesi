# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:34:06 2020

@author: aleca
"""


import os
import glob
import numpy as np
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from BayesianModel import *
from utils.queryExpansion import *
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
query,q_id,q_cams=query_feature[0:300],query_id[0:300],query_cams[0:300]


print('START')
ranks_index,ranks_sim,ranks_label = calculateRanks_Similarity(query,gallery,g_id,Bayes)
print('Ranks calcolato')

#Seleziono i campioni pos e neg
pos,neg=np.zeros_like(ranks_index),np.zeros_like(ranks_index)
k=25
for i in range(len(q_id)):
    pos[0:k,i]= ranks_label[0:k,i]==q_id[i]
    neg[0:k,i]= ranks_label[0:k,i]!=q_id[i]
    

Ranks_index=[]
vettori_cmc,vettore_mAP=[],[]

#Calcolo CMC
ranks_label=np.array(g_id)[ranks_index]
cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)
vettori_cmc.append(cmc_vector)

#Calcolo mAP
mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
vettore_mAP.append(mAP)

print(mAP,cmc_vector[0])
Ranks_index.append(ranks_index.copy())


risultati=[len(set(q_id)),q_id]
risultatiRanks=[len(set(q_id)),q_id]

#Matlab adatto la gallery
#gallery=matlab.double(gallery.tolist())
#Start matlab
eng = matlab.engine.start_matlab()
R=eng.test(matlab.double(pos.tolist()),matlab.double(neg.tolist()))
print('Ranks calcolato')    
    

#Calcolo CMC
R=np.array(R,dtype=int)-1
ranks_label=np.array(g_id)[R]
cmc_vector=calculateCmcFromRanks(R,ranks_label,q_id,g_cams,q_cams)
vettori_cmc.append(cmc_vector)

#Calcolo mAP
mAP=calculate_mAP(R,ranks_label,q_id,g_cams,q_cams)
vettore_mAP.append(mAP)

print(mAP,cmc_vector[0])
###################################àà

n,k=3,25
start=time.time()
for i in range(n):
    #Calcolo nuova query
    for i in range(len(q_id)):
        ranks_label=np.array(g_id)[ranks_index[:,i]]
        
        #Selezioni i primi k indici del rank
        top_index=ranks_index[:,i][0:k]
        
        #Selezioni gli indici dei campioni positive e negative
        positive=[top_index[j] for j in range(len(top_index)) if ranks_label[j]==q_id[i]]
        negative=[top_index[j] for j in range(len(top_index)) if ranks_label[j]!=q_id[i]]
        
        #EMR
        
        #all the elements are converted to matlab types, with particular attention to indexes
        positive = np.array(positive)+1 # no 0 index in matlab
        negative = np.array(negative)+1 # no 0 index in matlab
        
        scores, r_index = eng.EMR(matlab.double(query[i].tolist()),0,matlab.uint16(positive.tolist()), matlab.uint16(negative.tolist()), nargout=2)
#       score,r_index = eng.EMR(query[i], gallery, positive, negative)
        scores=np.array(scores)
        r_index=np.array(r_index)
        
        ranks_index[:,i]=r_index-1
 
    print('Ranks calcolato')
    #Calcolo CMC
    ranks_label=np.array(g_id)[ranks_index]
    cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)
    vettori_cmc.append(cmc_vector)
    
    #Calcolo mAP
    mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
    vettore_mAP.append(mAP)
    
    print(mAP,cmc_vector[0])

    Ranks_index.append(ranks_index.copy())

    
risultati.append([[k,n,vettori_cmc,vettore_mAP]])
risultatiRanks.append([[k,n,Ranks_index]])

end=time.time()
print('Tempo' + str(end-start))

f=open('Market_EMR.pkl','wb')
pickle.dump(risultati,f)
f.close()

f=open('Ranks-Market_EMR.pkl','wb')
pickle.dump(risultatiRanks,f)
f.close()


