# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:55:36 2020

@author: aleca
"""

import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from BayesianModel import *
from utils.queryExpansion import *
import pickle

#Test con 300 immagini per studiare le prestazioni al variare delle iterazioni e k.
#n=10 e k varia tra 5,15,25,35,45,55
#Niente feedback e per la probabilità la soglia è 0,5
#Dataset Market e Duke

#Scelta dataset
#Dataset='Duke'
#Dataset='Market'
def test300_BQE(Dataset):
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
     
    #Load BayesianModel gia addestrato
    Bayes=loadFile(DirBayes)
    
    print('TRAINING COMPLETE')
         
    gallery,gallery_id,gallery_cams=test_feature,test_id,test_cams
    query,query_id,query_cams=query_feature[0:300],query_id[0:300],query_cams[0:300]
    
    query_first=query
    
    print('START TEST')
    
    #Calcolo primo rank
    first_ranks_index,first_ranks_probability,first_ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
    
    #Calcolo la prima cmc
    first_cmc_vector=calculateCmcFromRanks(first_ranks_index,first_ranks_label,query_id,gallery_cams,query_cams)
    
    #Calcolo il primo mAP
    first_mAP=calculate_mAP(first_ranks_index,first_ranks_label,query_id,gallery_cams,query_cams)
    
    n=10
    results,results_Ranks=[],[]
    
    risultatiTest=[len(set(query_id)),query_id]
    risultatiTest_Ranks=[len(set(query_id)),query_id]
    for k in [5,15,25,35,45,55]: 
        query=query_first
        ranks_index,ranks_probability,ranks_label = first_ranks_index,first_ranks_probability,first_ranks_label 
        vettori_cmc,ranks,mAP_list=[],[],[] 
        
        ranks.append(first_ranks_index)
        vettori_cmc.append(first_cmc_vector)
        mAP_list.append(first_mAP)
        print(k)
        for i in range(n):
            query=queryExpansion(ranks_index,ranks_probability,gallery,query,k,soglia=0.5)
            print('Nuova query calcolata')
    
            ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
            print('Ranks calcolato')
            ranks.append(ranks_index)
            
            #Calcolo la cmc
            cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,query_id,gallery_cams,query_cams)
            vettori_cmc.append(cmc_vector)
        
            #Calcolo mAP
            mAP=calculate_mAP(ranks_index,ranks_label,query_id,gallery_cams,query_cams)
            mAP_list.append(mAP)
            
        k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
        results.append(k_n_cmc_mAP)
    
        k_n_ranks=[k,n,ranks]
        results_Ranks.append(k_n_ranks)
        
        print('####################')
    risultatiTest.append(results)  
    risultatiTest_Ranks.append(results_Ranks)
        
    f=open(Dataset + '_300pics_k_n_BQE.pkl','wb')
    pickle.dump(risultatiTest,f)
    f.close()
    
    f=open('Ranks-'+ Dataset + '_300pics_k_n_BQE.pkl','wb')
    pickle.dump(risultatiTest_Ranks,f)
    f.close()
