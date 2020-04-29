# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:13:17 2020

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
from cmc import *
from Lbp import *
from hog import *
from BayesianModel import *
import time
import random

def calculateRanks2(query,gallery,g_id,B):
    ranks_index=np.zeros((len(gallery),len(query)),int)
    ranks_label=np.zeros((len(gallery),len(query)),int)
    ranks_similarity=np.zeros((len(gallery),len(query)),float)
    column=0
    for q in query:
        d=np.zeros(len(gallery))
        s=np.zeros(len(gallery))
        i=0
        for g in gallery:
            d[i]=histogram_distance(q,g)
            #s[i]=histogram_intersection(q,g)
            i+= 1
        sorted_i=np.argsort(d)
        ranks_index[:,column] = sorted_i
        ranks_label[:,column] = [g_id[i] for i in sorted_i]

        ranks_similarity[:,column]= np.sort(d)
        column += 1
    return ranks_index,ranks_similarity,ranks_label 

def queryExpansion2(ranks_index,ranks_similarity,gallery,query,Bayes,K):
    q_expansion=[]
    for i in range(len(query)):
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_similarity=ranks_similarity[:,i][0:K] #Prendo il prime K distanze del rank
        q_exp=0 
        similarity_sum=0
        for j in range(K):
            similarity=candidates_similarity[j]
            x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
            q_exp += similarity*x
            similarity_sum += similarity 
        q_exp=(q_exp +query[i])/(similarity_sum +1)
        q_expansion.append(q_exp)    
    return q_expansion


if __name__ == '__main__':
    
    DirMarket = '..\\FeatureCNN\\Market-1501'
    DirDuke = '..\\FeatureCNN\\DukeMTMC'
    
    
    testData,queryData,trainingData=loadCNN(DirMarket)
    #testData,queryData,trainingData=loadMarket_1501(feature=True)
    
    test_cams, test_feature, test_id, test_desc = testData
    query_cams, query_feature, query_id, query_desc = queryData
    train_cams, train_feature, train_id, train_desc = trainingData
    
    #Load BayesianModel gia addestrato
    B=loadFile('..\\B_Market_trained')
    #B=loadFile('..\\B_Duke_trained.pkl')
    #B.plotTrainingHistogram(True)
    
    print('TRAINING COMPLETE')
    
    gallery,gallery_id=test_feature,test_id
    query,query_id = query_feature[0:30:5], query_id[0:30:5]
    print('START TEST')
       
    start=time.time()
    vettori_cmc,ranks=[],[]    
    for i in range(2):
        ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,B)
        print('Ranks calcolato')
        ranks.append(ranks_label)
        
        #Calcolo la cmc
        cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
        
        #Calcolo mAP
        mAP=calculate_mAP(ranks_label,query_id,100)
        
        vettori_cmc.append(cmc_vector)
        r1,r5,r10,r20,r50=cmc_vector[[0,4,9,19,49]]
        print('Rank 1: {} - Rank 5: {} - Rank 10: {} - Rank 20: {} - Rank 50: {}  '.format(r1,r5,r10,r20,r50))
        print('mAP: {}'.format(mAP))
        
        query=queryExpansion(ranks_index,ranks_probability,gallery,query,5) 
        print('Nuova query calcolata')
    
    end=time.time()
    tempo=end-start
    print(tempo)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
