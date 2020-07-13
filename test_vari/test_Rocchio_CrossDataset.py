# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:34:13 2020

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
from ROCCHIO import *
import time
import random
import pickle

#Test Rocchio con n=3 e k=55 cross dataset.

#Scelta dataset
#Dataset='DukeFromMarket'
#Dataset='MarketFromDuke' 
def test_Rocchio(Dataset):
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

    #Feature CNN
    testData,queryData,trainingData=loadCNN(DirCNN)
    
    Bayes=loadFile(DirBayes)
    
    test_cams, test_feature, test_id, test_desc = testData
    query_cams, query_feature, query_id, query_desc = queryData
    train_cams, train_feature, train_id, train_desc = trainingData
    
    #Cross dataset
    f=open(DirCross,'rb')
    test_feature, query_feature=pickle.load(f)
    f.close()
    

    gallery,g_id,g_cams=test_feature,test_id,test_cams
    query,q_id,q_cams=query_feature[0::],query_id[0::],query_cams[0::]
    
    
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
            distances, r_index = query_shift(query[i],gallery,positive,negative)
            ranks_index[:,i]=r_index
            
        print('Ranks calcolato')
        
    risultati.append([[k,n,vettori_cmc,vettore_mAP]])
    risultatiRanks.append([[k,n,Ranks_index]])
    
    
    f=open(Dataset + '_Rocchio.pkl','wb')
    pickle.dump(risultati,f)
    f.close()
    
    f=open('Ranks-'+ Dataset +'_Rocchio.pkl','wb')
    pickle.dump(risultatiRanks,f)
    f.close()
    
    
    #for ranks_index in Ranks_index:
    #    ranks_label=np.array(g_id)[ranks_index]
    #    m=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
    #    r1=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)[0]
    #    print((m,r1))