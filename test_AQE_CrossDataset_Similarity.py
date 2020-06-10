# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:51:01 2020

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


#Test AQE con n=3 e k=5 cross dataset. Usata similarità.

#Scelta dataset
#Dataset='DukeFromMarket'
#Dataset='MarketFromDuke'
def test_AQE_CrossDataset_Similarity(Dataset):
    if Dataset == 'DukeFromMarket':
        DirCNN= '..\\FeatureCNN\\DukeMTMC'
        #Cross Dataset
        DirCross='..\\FeatureCNN\\CrossDataset\\DukeFromMarket_feature.pkl'
        DirBayes='..\\Bayes_Market_trained.pkl'
        
    elif Dataset == 'MarketFromDuke':
        DirCNN= '..\\FeatureCNN\\Market-1501'
        #Cross Dataset
        DirCross='..\\FeatureCNN\\CrossDataset\\MarketFromDuke_feature.pkl'
        DirBayes='..\\Bayes_Duke_trained.pkl'
    else:
        print('ERRORE')
    
    print('START')
    #Feature CNN
    testData,queryData,trainingData=loadCNN(DirCNN)
    
    #Load BayesianModel gia addestrato
    Bayes=loadFile(DirBayes)
    
    test_cams, test_feature, test_id, test_desc = testData
    query_cams, query_feature, query_id, query_desc = queryData
    train_cams, train_feature, train_id, train_desc = trainingData
    
    #Cross dataset
    f=open(DirCross,'rb')
    test_feature, query_feature=pickle.load(f)
    f.close()
    
    gallery,gallery_id=test_feature,test_id
    query,query_id = query_feature[0::], query_id[0::]
    
            
    start=time.time()
    
    
    print('START TEST')
    
    n,k=3,5
    vettori_cmc,ranks,mAP_list=[],[],[] 
    for i in range(n+1):
        ranks_index,ranks_probability,ranks_label =calculateRanks_Similarity(query,gallery,gallery_id,Bayes)
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
    
    
        
    f=open(Dataset + '_test_complete_AQE_Similarity.pkl','wb') 
    pickle.dump(results,f)
    f.close()
    
    f=open('Ranks-' + Dataset + '_test_complete_AQE_Similarity.pkl','wb') 
    pickle.dump(results_ranks,f)
    f.close()
    
    print('Fine')