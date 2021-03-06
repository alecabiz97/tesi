# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:32:36 2020

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

#Test feedback non pesato con n=3 e k=55. Usata la similarità

#Scelta dataset
#Dataset='Duke'
#Dataset='Market'
def test_FeedbackNonPesato_Similarity(Dataset):
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
    
    gallery,gallery_id,gallery_cams=test_feature,test_id,test_cams
    query,query_id,query_cams = query_feature[0::], query_id[0::], query_cams[0::]

    
    
    start=time.time()
    
    
    print('START TEST')
    print(len(set(query_id)))
    
    n,k=3,55
    vettori_cmc,ranks,mAP_list=[],[],[] 
    for i in range(n+1):
        ranks_index,ranks_probability,ranks_label =calculateRanks_Similarity(query,gallery,gallery_id,Bayes)
        ranks.append(ranks_index)
        print('Ranks calcolato')
        
        #Calcolo la cmc
        cmc_vector=calculateCmcFromRanks(ranks_index,ranks_label,query_id,gallery_cams,query_cams)
        vettori_cmc.append(cmc_vector)
    
        #Calcolo mAP
        mAP=calculate_mAP(ranks_index,ranks_label,query_id,gallery_cams,query_cams)
        mAP_list.append(mAP)
    
        #Calcolo la nuova query simulando il feedback da parte di un utente
        query=queryExpansion_withFeedback_CameraFilter(ranks_index,ranks_probability,ranks_label,gallery,gallery_cams,query,query_id,query_cams,k,True)
        print('Nuova query calcolata')
        
    
    #Senza i ranks
    results=[len(set(query_id)),query_id]
    k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
    results.append([k_n_cmc_mAP])
    
    #Solo i ranks
    results_ranks=[len(set(query_id)),query_id]
    k_n_ranks=[k,n,ranks]
    results_ranks.append([k_n_ranks])
    
    f=open(Dataset + '_test_complete_Feedback_NonPesato_Similarity_CameraFilter.pkl','wb') 
    pickle.dump(results,f)
    f.close()
    
    f=open('Ranks-' + Dataset +'_test_complete_Feedback_NonPesato_Similarity_CameraFilter.pkl','wb') 
    pickle.dump(results_ranks,f)
    f.close()
    
    print('Fine')