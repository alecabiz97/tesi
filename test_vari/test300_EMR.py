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
import time
import random
import pickle
import matlab.engine

#test con 300 immagini per studiare mAP e cmc al variare di k e n. EME

#Dataset='Duke'
#Dataset='Market'    
def test300_EMR(Dataset):
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
    
    #Calcolo CMC
    first_cmc_vector=calculateCmcFromRanks(first_ranks_index,first_ranks_label,q_id,g_cams,q_cams)
    
    #Calcolo mAP
    first_mAP=calculate_mAP(first_ranks_index,first_ranks_label,q_id,g_cams,q_cams)

    
    #Strart Matlab
    eng = matlab.engine.start_matlab()

    results=[]
    resultsRanks=[]
    n=1
    for k in [5,15,25,35,45,55]:    
        print(k)
        ranks_index=first_ranks_index.copy()
        Ranks_index=[ranks_index]
        vettori_cmc,vettore_mAP=[first_cmc_vector],[first_mAP]
    
        for i in range(n):
            start=time.time()
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
#                scores=np.array(scores)
                r_index=np.array(r_index)
                ranks_index[:,i]=r_index-1
            print('Ranks calcolato')
            end=time.time()
            print('Tempo' + str(end-start))
            
            #Calcolo CMC
            ranks_label=np.array(g_id)[ranks_index]
            cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,q_id,g_cams,q_cams)
            vettori_cmc.append(cmc_vector)
            
            #Calcolo mAP
            mAP=calculate_mAP(ranks_index,ranks_label,q_id,g_cams,q_cams)
            vettore_mAP.append(mAP)
            
            Ranks_index.append(ranks_index.copy())
        
        
        results.append([k,n,vettori_cmc,vettore_mAP])
        resultsRanks.append([k,n,Ranks_index])
    
    risultatiTest=[len(set(q_id)),q_id,results]
    risultatiRanks=[len(set(q_id)),q_id,resultsRanks]

    
    f=open(Dataset + '300_EMR.pkl','wb')
    pickle.dump(risultatiTest,f)
    f.close()
    
    f=open('Ranks-' + Dataset + '300_EMR.pkl','wb')
    pickle.dump(risultatiRanks,f)
    f.close()
    return risultatiTest,risultatiRanks

A,B=test300_EMR('Market')    
  
