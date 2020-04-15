# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:28:01 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importData import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
import time
import random


class BayesianModel(object):
    
    def __init__(self,K=10):
        self.K=K
        
    def train(self,train,labels,M=100):
        #Stimo P(lti==ltj) e P(lti!=ltj) 
        n_id=len(set(labels))
        labels_set=list(set(labels))
        P=np.zeros(n_id)
        i=0
        for label in labels_set:
            P[i]=(1/(len(train)-1))*(labels.count(label)-1)
            i += 1
        self.P_ltiEqualsltj=np.average(P)
        self.P_ltiNotEqualsltj = 1 - self.P_ltiEqualsltj
            
        
        d_sameId,d_differentId=[],[]
        i=0
        for xi,label_xi in zip(train,labels):
            for xj,label_xj in zip(train[i::],labels[i::]):
                if not np.array_equal(xi,xj):
                    d=(1/(1+histogram_intersection(xi,xj)))  #Calcolo distanza
                    if label_xi == label_xj:
                        d_sameId.append(d)
                    else:
                        d_differentId.append(d)
            i += 1 
                        
        d_sameId,d_differentId=np.array(d_sameId),np.array(d_differentId)
        
        
        #np.histogram() restituisce una lista che contiene l'istgramma e un vettore con gli estremi dei bins
        self.hist_d_sameId=np.histogram(d_sameId,M) 
        self.hist_d_differentId=np.histogram(d_differentId,M) 
        
        
    def calculateP_d(self,distanza):
        h_sameId, bins_sameId = self.hist_d_sameId
        h_diffId, bins_diffId = self.hist_d_differenId
        
        #Calcolo P_d_lqEqualslg
        if distanza < np.min(bins_sameId):
            distanza_index=0
        else:    
            distanza_index=np.sum(distanza>bins_sameId)-1 #è l'indice dell'intervallo in cui cade distanza
         
        P_d_lqEqualslg = h_sameId[distanza_index]/np.sum(h_sameId)
        
        #Calcolo P_d_lqNotEqualslg
        if distanza < np.min(bins_diffId):
            distanza_index=0
        else:
            distanza_index=np.sum(distanza>bins_diffId)-1 #è l'indice dell'intervallo dell'istogramma in cui cade distanza

        P_d_lqNotEqualslg = h_diffId[distanza_index]/np.sum(h_diffId)
        
        P_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)+(P_d_lqNotEqualslg*self.P_ltiNotEqualsltj)

        #Teorema di Bayes
        P_lqEqualslg_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)/P_d
        return P_lqEqualslg_d
     


def calculateRanks(query,gallery,Bayes):
    ranks=np.zeros((len(gallery),len(query)),int)
    ranks_probability=np.zeros((len(gallery),len(query)),float)
    column=0
    for q in query:
        d=np.zeros(len(gallery))
        p=np.zeros(len(gallery))
        i=0
        for g in gallery:
            d[i]=(1/(1+histogram_intersection(q,g)))
            p[i]=Bayes.calculateP_d(d[i])
            i+= 1
        sorted_i=np.argsort(d)
        ranks[:,column] = sorted_i
        ranks_probability[:,column]= -np.sort(-p)
        column += 1
    return ranks,ranks_probability 

def queryExpansion(ranking,gallery,query,K):
    ranks_index,ranks_probability=ranking[0],ranking[1]
    q_expansion=[]
    for i in range(len(query)):
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        q_exp=query[i] #Query originale
        probability_sum=0
        for j in range(K):
            probability_c=candidates_probability[j]
            c_hist=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
            q_exp=q_exp + probability_c*c_hist
            probability_sum += probability_c +1
        q_exp=q_exp/probability_sum
        q_expansion.append(q_exp)    
    return q_expansion
                
if __name__ == '__main__':
    

    print('START TRAINING')

    B=BayesianModel()
    B.train(hist_of_train,labels)
    print('TRAINING COMPLETE')
    print((B.P_ltiEqualsltj,B.P_ltiNotEqualsltj)) 
    
    pl.plot(B.hist_d_sameId[0])
    pl.plot(B.hist_d_differenId[0]) 

    
 
        
                
                
                
                
                
                
                
                
                
                
        