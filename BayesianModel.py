# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:28:01 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importBmp import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
import time
import random


class BayesianModel(object):
    
    def __init__(self,K=5):
        self.K=K
        
    def train(self,gallery_train,gallery_labels,query_train,query_labels):
        gallery_labels=np.array(gallery_labels)
        query_labels=np.array(query_labels)
        
        ranks=np.zeros((len(gallery_train),len(query_train)))
        column=0
        for q in query_train:
            d=np.zeros(len(gallery_train))
            i=0
            for h in gallery_train:
                    d[i]=(1/(1+histogram_intersection(q,h)))
                    i+= 1
            sorted_id=np.argsort(d)
            sorted_labels=gallery_labels[sorted_id]
            ranks[:,column] = sorted_labels  #nella colonna ho il ranking con gli id 
            column += 1
    
        label_found=0
        i=0
        for label in query_labels:
            label_found += np.sum(ranks[0:self.K,i]==label)
            i += 1
    
                       
        self.P_ltiEqualsltj=label_found/len(query_labels)
        self.P_ltiNotEqualsltj= 1-self.P_ltiEqualsltj
        
    def test(self,query,gallery):        
        ranks_distance=np.zeros((len(gallery),len(query)))
        ranks_index=np.zeros_like(ranks_distance,int)
        column=0
        for q in query:
            d=np.zeros(len(gallery))
            i=0
            for g in gallery:
                d[i]=(1/(1+histogram_intersection(q,g)))
                i+= 1
            sorted_i=np.argsort(d)
            ranks_index[:,column] = sorted_i
            ranks_distance[:,column] = np.sort(d)
            column += 1
        self.distance=ranks_distance
        return [ranks_index,ranks_distance] 
        
        
    def calculateP_d(self,distanza,M=10):
        h,bins=np.histogram(self.distance,M)
        
        for i in range(len(bins)):
             if distanza <= bins[i] or distanza == np.min(self.distance):
                 break
         
        P_d_lqEqualslg=h[i-1]/np.sum(h)
        P_d_lqNotEqualslg= 1 - P_d_lqEqualslg
        
        P_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)+(P_d_lqNotEqualslg*self.P_ltiNotEqualsltj)

        #Teorema di Bayes
        P_lqEqualslg_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)/P_d
        return P_lqEqualslg_d
     
        
def ranking(query,gallery):
    ranks_distance=np.zeros((len(gallery),len(query)))
    ranks_index=np.zeros_like(ranks_distance,int)
    column=0
    for q in query:
        d=np.zeros(len(gallery))
        i=0
        for h in gallery:
                d[i]=(1/(1+histogram_intersection(q,h)))
                i+= 1
        sorted_i=np.argsort(d)
        ranks_index[:,column] = sorted_i
        ranks_distance[:,column] = np.sort(d)
        column += 1
    return [ranks_index,ranks_distance]    
    
def queryExpansion(ranking,test,query,Bayes,K):
    ranks_index,ranks_distance=ranking[0],ranking[1]
    q_expansion=[]
    for q,i in zip(query,range(ranks_index.shape[1])):
        candidates_i=ranks_index[:,i][0:K]
        candidates_distance=ranks_distance[:,i][0:K]
        q_exp=q 
        probability_sum=0
        for j in range(K):
            probability_c=Bayes.calculateP_d(candidates_distance[j])
            c_hist=test[candidates_i[j]]
            q_exp=q_exp + probability_c*c_hist
            probability_sum += probability_c +1
        q_exp=q_exp/probability_sum
        q_expansion.append(q_exp)    
    return q_expansion


                
if __name__ == '__main__':
    
#    gallery=[histogram_vector(camA[i]) for i in range(10)]
#    query=[histogram_vector(camB[i]) for i in range(10)]
#    
#    r1,r2=ranking(query,gallery)
    
         

    random_id=np.random.permutation(np.arange(0,len(camA[0:50])))
    train_index,test_index=np.split(random_id,2)
    
    
    gallery_train=[histogram_vector(camA[i]) for i in train_index]
    query_train=[histogram_vector(camB[i]) for i in train_index]  
    gallery_train_labels=[Id_A[i] for i in train_index]
    query_train_labels=[Id_B[i] for i in train_index]
    
    gallery_test=[histogram_vector(camA[i]) for i in test_index]
    query_test=[histogram_vector(camB[i]) for i in test_index]

    

    B=BayesianModel()
    B.train(gallery_train,gallery_train_labels,query_train,query_train_labels) 
    
    
    r=[]
    for i in range(2):
        rank=B.test(query_test,gallery_test)
        query_test=queryExpansion(rank,gallery_test,query_test,B,20) 
        r.append(rank[0])
                
                
                
                
                
                
                
                
                
                
        