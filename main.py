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
testData,queryData,trainingData=loadCNN(DirDuke)

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

gallery,g_id,g_cams=test_feature[0:500],test_id[0:500],test_cams[0:500]
query,q_id,q_cams=query_feature[0:1],query_id[0:1],query_cams[0:1]



Ranks_index=[]


print('START')
ranks_index,ranks_sim,ranks_label = calculateRanks_Similarity(query,gallery,g_id,Bayes)
print('Ranks calcolato')

risultati=[len(set(q_id)),q_id]
risultatiRanks=[len(set(q_id)),q_id]

n,k=3,55
vettori_cmc,vettore_mAP=[],[]
for i in range(n+1):
    
    #Calcolo CMC
    ranks_label=np.array(g_id)[ranks_index]
    cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)
    vettori_cmc.append(cmc_vector)
    
    #Calcolo mAP
    mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
    vettore_mAP.append(mAP)
    
    Ranks_index.append(ranks_index.copy())

    #Calcolo nuova query
    for i in range(len(q_id)):
        ranks_label=np.array(g_id)[ranks_index[:,i]]
        
        #Selezioni i primi k indici del rank
        top_index=ranks_index[:,i][0:k]
        
        #Selezioni gli indici dei campioni positive e negative
        positive=[top_index[j] for j in range(len(top_index)) if ranks_label[j]==q_id[i]]
        negative=[top_index[j] for j in range(len(top_index)) if ranks_label[j]!=q_id[i]]
            
        #Rocchio
        #distances, r_index = query_shift(query[i],gallery,positive,negative)
        
        #EMR
        eng = matlab.engine.start_matlab()
        #all the elements are converted to matlab types, with particular attention to indexes
        positive = np.array(positive)+1 # no 0 index in matlab
        negative = np.array(negative)+1 # no 0 index in matlab
        
        scores, r_index = eng.EMR(matlab.double(query[i].tolist()), matlab.double(gallery.tolist()), matlab.uint16(positive.tolist()), matlab.uint16(negative.tolist()), nargout=2)
#       score,r_index = eng.EMR(query[i], gallery, positive, negative)
        print(1)
        ranks_index[:,i]=r_index
    print('Ranks calcolato')
    
risultati.append([[k,n,vettori_cmc,vettore_mAP]])
risultatiRanks.append([[k,n,Ranks_index]])


#f=open('DukeFromMarket_Rocchio.pkl','wb')
#pickle.dump(risultati,f)
#f.close()
#
#f=open('Ranks-DukeFromMarket_Rocchio.pkl','wb')
#pickle.dump(risultatiRanks,f)
#f.close()


for ranks_index in Ranks_index:
    ranks_label=np.array(g_id)[ranks_index]
    m=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
    r1=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)[0]
    print((m,r1))




    





#r1,r2,r3=calculateRanks_Similarity(query,gallery,g_id,Bayes)
#x1=calculateCmcFromRanks(r3,q_id)
#m1=calculate_mAP(r3,q_id,len(r3))




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