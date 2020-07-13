# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:42:00 2020

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
import matlab.engine

#test con 300 immagini per studiare mAP e cmc al variare di k e n. Rocchio

#Dataset='Duke'
#Dataset='Market'    
def test300_Rocchio(Dataset):
    if Dataset == 'Duke':
        DirCNN= '..\\FeatureCNN\\DukeMTMC'
        DirBayes='..\\Bayes_Duke_trained.pkl'
    elif Dataset == 'Market':
        DirCNN= '..\\FeatureCNN\\Market-1501'
        DirBayes='..\\Bayes_Market_trained.pkl'
    else:
        print('ERRORE')
    
    print('START')
    
    #Feature CNN
    testData,queryData,trainingData=loadCNN(DirCNN)
    
    test_cams, test_feature, test_id, test_desc = testData
    query_cams, query_feature, query_id, query_desc = queryData
    train_cams, train_feature, train_id, train_desc = trainingData

    gallery,g_id,g_cams=test_feature,test_id,test_cams
    query,q_id,q_cams=query_feature[0:300],query_id[0:300],query_cams[0:300]
    
    Bayes='Non serve'
    print('START')
    first_ranks_index,first_ranks_sim,first_ranks_label = calculateRanks_Similarity(query,gallery,g_id,Bayes)
    print('Ranks calcolato')
    
    
    results=[]
    resultsRanks=[]
    n=10
    for k in [5,15,25,35,45,55]:
        print(k)
        ranks_index=first_ranks_index.copy()
        Ranks_index=[]
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
        
        results.append([k,n,vettori_cmc,vettore_mAP])
        resultsRanks.append([k,n,Ranks_index])
    
    risultatiTest=[len(set(q_id)),q_id,results]
    risultatiRanks=[len(set(q_id)),q_id,resultsRanks]
    
    f=open(Dataset + '300_Rocchio.pkl','wb')
    pickle.dump(risultatiTest,f)
    f.close()
    
    f=open('Ranks-' + Dataset + '300_Rocchio.pkl','wb')
    pickle.dump(risultatiRanks,f)
    f.close()

    
  