# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:42:09 2020

@author: aleca
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

#AQE

DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

#Cross dataset
DirMarketFromDuke='..\\FeatureCNN\\CrossDataset\\MarketFromDuke_feature.pkl'
DirDukeFromMarket='..\\FeatureCNN\\CrossDataset\\DukeFromMarket_feature.pkl'
f=open(DirMarketFromDuke,'rb')
test_feature, query_feature=pickle.load(f)
f.close()


#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
Bayes=loadFile('..\\Bayes_Duke_trained.pkl')


gallery,gallery_id=test_feature,test_id
query,query_id = query_feature[0::], query_id[0::]

        
start=time.time()


print('START TEST')

n,k=3,5
vettori_cmc,ranks,mAP_list=[],[],[] 
for i in range(n+1):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
    ranks.append(ranks_label)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    
    
    #Calcolo la nuova query    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,k,AQE=True)
    print('Nuova query calcolata')
    

#Cmc e mAP
results=[len(set(query_id)),query_id]
k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
results.append([k_n_cmc_mAP])

#Solo i ranks
results_ranks=[len(set(query_id)),query_id]
k_n_ranks=[k,n,ranks]
results_ranks.append([k_n_ranks])


    
f=open('MarketFromDuke_test_complete_AQE.pkl','wb') 
pickle.dump(results,f)
f.close()

f=open('Ranks-MarketFromDuke_test_complete_AQE.pkl','wb') 
pickle.dump(results_ranks,f)
f.close()
#######################################################################################
#####################################################################################
#BQE
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)

test_cams, test_feature, test_ids, test_desc = testData
query_cams, query_feature, query_ids, query_desc = queryData
train_cams, train_feature, train_ids, train_desc = trainingData
    
#Cross dataset
DirMarketFromDuke='..\\FeatureCNN\\CrossDataset\\MarketFromDuke_feature.pkl'
DirDukeFromMarket='..\\FeatureCNN\\CrossDataset\\DukeFromMarket_feature.pkl'
f=open(DirMarketFromDuke,'rb')
test_feature, query_feature=pickle.load(f)
f.close()


#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
Bayes=loadFile('..\\Bayes_Duke_trained.pkl')


gallery,gallery_id=test_feature,test_ids
query,query_id = query_feature[0::], query_ids[0::]

        
start=time.time()


print('START TEST')

n,k=3,5
vettori_cmc,ranks,mAP_list=[],[],[] 
for i in range(n+1):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
    ranks.append(ranks_label)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    
    
    #Calcolo la nuova query    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,k,AQE=False,soglia=0.5)
    print('Nuova query calcolata')
    

#Cmc e mAP
results=[len(set(query_id)),query_id]
k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
results.append([k_n_cmc_mAP])

#Solo i ranks
results_ranks=[len(set(query_id)),query_id]
k_n_ranks=[k,n,ranks]
results_ranks.append([k_n_ranks])


    
f=open('MarketFromDuke_test_complete_soglia0,5.pkl','wb') 
pickle.dump(results,f)
f.close()

f=open('Ranks-MarketFromDuke_test_complete_soglia0,5.pkl','wb') 
pickle.dump(results_ranks,f)
f.close()
#######################################################################################
#####################################################################################
#FEEDBACK  PESATO

print('START')

    
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

#Cross dataset
DirMarketFromDuke='..\\FeatureCNN\\CrossDataset\\MarketFromDuke_feature.pkl'
DirDukeFromMarket='..\\FeatureCNN\\CrossDataset\\DukeFromMarket_feature.pkl'
f=open(DirMarketFromDuke,'rb')
test_feature, query_feature=pickle.load(f)
f.close()

    
#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
Bayes=loadFile('..\\Bayes_Duke_trained.pkl')

Bayes.plotTrainingHistogram(True)
d=np.arange(0,3,0.05)
s=[1/(1+i) for i in d]
p=[Bayes.calculateProbBayes(i) for i in d]

pl.plot(d,s,label='Similarità')
pl.plot(d,p,label='Probabilità sameId')
pl.legend()
pl.grid()
pl.show()

gallery,gallery_id=test_feature,test_id
query,query_id = query_feature[0::], query_id[0::]


start=time.time()


print('START TEST')
print(len(set(query_id)))

n,k=3,55
vettori_cmc,ranks,mAP_list=[],[],[] 
for i in range(n+1):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
    ranks.append(ranks_label)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    

    #Calcolo la nuova query simulando il feedback da parte di un utente
    query=queryExpansion_withFeedback(ranks_index,ranks_probability,ranks_label,gallery,query,query_id,k,False)
    print('Nuova query calcolata')
    

#Senza i ranks
results=[len(set(query_id)),query_id]
k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
results.append([k_n_cmc_mAP])

#Solo i ranks
results_ranks=[len(set(query_id)),query_id]
k_n_ranks=[k,n,ranks]
results_ranks.append([k_n_ranks])



f=open('MarketFromDuke_test_complete_HumanFeedback_Prob_k55.pkl','wb') 
pickle.dump(results,f)
f.close()

f=open('Ranks-MarketFromDuke_test_complete_HumanFeedback_Prob_k55.pkl','wb') 
pickle.dump(results_ranks,f)
f.close()

#######################################################################################
#####################################################################################
#FEEDBACK NON PESATO
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirMarket)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

#Cross dataset
DirMarketFromDuke='..\\FeatureCNN\\CrossDataset\\MarketFromDuke_feature.pkl'
DirDukeFromMarket='..\\FeatureCNN\\CrossDataset\\DukeFromMarket_feature.pkl'
f=open(DirMarketFromDuke,'rb')
test_feature, query_feature=pickle.load(f)
f.close()

    
#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
Bayes=loadFile('..\\Bayes_Duke_trained.pkl')


gallery,gallery_id=test_feature,test_id
query,query_id = query_feature[0::], query_id[0::]


start=time.time()


print('START TEST')
print(len(set(query_id)))

n,k=3,55
vettori_cmc,ranks,mAP_list=[],[],[] 
for i in range(n+1):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
    ranks.append(ranks_label)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    

    #Calcolo la nuova query simulando il feedback da parte di un utente
    query=queryExpansion_withFeedback(ranks_index,ranks_probability,ranks_label,gallery,query,query_id,k,True)
    print('Nuova query calcolata')
    

#Senza i ranks
results=[len(set(query_id)),query_id]
k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
results.append([k_n_cmc_mAP])

#Solo i ranks
results_ranks=[len(set(query_id)),query_id]
k_n_ranks=[k,n,ranks]
results_ranks.append([k_n_ranks])



f=open('MarketFromDuke_test_complete_HumanFeedback_Prob1_k55.pkl','wb') 
pickle.dump(results,f)
f.close()

f=open('Ranks-MarketFromDuke_test_complete_HumanFeedback_Prob1_k55.pkl','wb') 
pickle.dump(results_ranks,f)
f.close()

#######################################################################################
#####################################################################################

