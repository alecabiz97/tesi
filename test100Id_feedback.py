# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:44:20 2020

@author: AleCabiz
"""

import os
import glob
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importData import *
from histogram import *
from evaluation import *
from Lbp import *
from hog import *
from BayesianModel import *
from queryExpansion import *

import time
import random
import pickle


print('START')

    
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirDuke)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

    
#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
Bayes=loadFile('..\\Bayes_Duke_trained.pkl')

print('TRAINING COMPLETE')


#Seleziono 100 identità a caso
labels=np.random.permutation(list(set(query_id)))[0:100]
    
        
query_first=[query_feature[i] for i in range(len(query_id)) if query_id[i] in labels]  
query_ids=[query_id[i] for i in range(len(query_id)) if query_id[i] in labels]  
   
start=time.time()
    
gallery,gallery_id=test_feature,test_id
query,query_id=query_first,query_ids 


print('START TEST')


#Calcolo primo rank
first_ranks_index,first_ranks_probability,first_ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)

#Calcolo la prima cmc
first_cmc_vector=calculateCmcFromRanks(first_ranks_label,query_id)

#Calcolo il primo mAP
first_mAP=calculate_mAP(first_ranks_label,query_id,len(first_ranks_label))

n=10
results,results_Ranks=[],[]

risultatiTest=[len(labels),query_id]
risultatiTest_Ranks=[len(labels),query_id]
for k in [5,10,15,25,35,45,55]: 
    query,query_id=query_first,query_ids 
    ranks_index,ranks_probability,ranks_label = first_ranks_index,first_ranks_probability,first_ranks_label 
    vettori_cmc,ranks,mAP_list=[],[],[] 
    
    ranks.append(first_ranks_label)
    vettori_cmc.append(first_cmc_vector)
    mAP_list.append(first_mAP)
    print(k)
    for i in range(n):
        query=queryExpansion_withFeedback(ranks_index,ranks_probability,ranks_label,gallery,query,query_id,k,True)
        print('Nuova query calcolata')

        ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
        print('Ranks calcolato')
        ranks.append(ranks_label)
        
        #Calcolo la cmc
        cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
        vettori_cmc.append(cmc_vector)
    
        #Calcolo mAP
        mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
        mAP_list.append(mAP)

    k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
    results.append(k_n_cmc_mAP)
    k_n_ranks=[k,n,ranks]
    results_Ranks.append(k_n_ranks)

    
    print('####################à')
risultatiTest.append(results)  
risultatiTest_Ranks.append(results_Ranks)
    
f=open('Ranks-Duke_results_100Id_HumanFeedback_Prob1.pkl','wb')
pickle.dump(risultatiTest,f)
f.close()

f=open('Duke_results_100Id_HumanFeedback_Prob1.pkl','wb')
pickle.dump(risultatiTest_withoutRanks,f)
f.close()

    
end=time.time()
tempo=end-start
print(tempo)