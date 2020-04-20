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
        
        for i in range(len(train)):
            for j in range(len(train)):
                if j > i:
                    d=histogram_distance(train[i],train[j])
                    #d=histogram_intersection(train[i],train[j])
                    if labels[i] == labels[j]:
                        d_sameId.append(d)
                    else:
                        d_differentId.append(d)
                       
        d_sameId,d_differentId=np.array(d_sameId),np.array(d_differentId)
        
        #np.histogram() restituisce una lista che contiene l'istogramma e un vettore con gli estremi degli intervalli
        hist_sameId,bins1=np.histogram(d_sameId,M)
        hist_differentId,bins2=np.histogram(d_differentId,M)
        
        #Calcolo probabilità 
        hist_sameId=hist_sameId/np.sum(hist_sameId) 
        hist_differentId=hist_differentId/np.sum(hist_differentId)
        
        self.hist_d_sameId=[hist_sameId,bins1]
        self.hist_d_differentId=[hist_differentId,bins2]
        
        
    def calculateProbBayes(self,distanza):
        h_sameId, bins_sameId = self.hist_d_sameId
        h_diffId, bins_diffId = self.hist_d_differentId
        
        #Calcolo P_d_lqEqualslg
        if distanza >= np.min(bins_sameId) and distanza <= np.max(bins_sameId):
            distanza_index=np.sum(distanza>bins_sameId)-1 #è l'indice dell'intervallo in cui cade distanza
        elif distanza < np.min(bins_sameId):
            distanza_index=0
        elif distanza > np.max(bins_sameId):
            distanza_index=len(h_sameId) - 1    
         
        P_d_lqEqualslg = h_sameId[distanza_index]
        
        #Calcolo P_d_lqNotEqualslg
        if distanza >= np.min(bins_diffId) and distanza <= np.max(bins_diffId):    
            distanza_index=np.sum(distanza>bins_diffId)-1 #è l'indice dell'intervallo dell'istogramma in cui cade distanza
        elif distanza < np.min(bins_diffId):
            distanza_index=0
        elif distanza > np.max(bins_diffId):
            distanza_index=len(h_diffId) -1 

        P_d_lqNotEqualslg = h_diffId[distanza_index]
        
        P_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)+(P_d_lqNotEqualslg*self.P_ltiNotEqualsltj)

        #Teorema di Bayes
        P_lqEqualslg_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)/P_d
        return P_lqEqualslg_d
    
    def plotTrainingHistogram(self):
        h1,b1=self.hist_d_sameId
        h2,b2=self.hist_d_differentId
        
        x1=np.zeros_like(h1)
        x2=np.zeros_like(h2)
        for i in range(len(b1)-1):
            x1[i]=(b1[i] + b1[i+1])/2
            x2[i]=(b2[i] + b2[i+1])/2
        
        width_binsSame=(max(b1)-min(b1))/100
        width_binsDiff=(max(b2)-min(b2))/100
        
        pl.bar(x1,h1,width_binsSame,label='sameId',color='r')
        pl.bar(x2,h2,width_binsDiff,label='differentId',color='b')
        #pl.plot(x1,h1,label='sameId',color='r')
        #pl.plot(x2,h2,label='differentId',color='b')
         
        pl.legend()
        pl.xlabel('Distance')
        pl.ylabel('Probability')
        pl.show()
        
        return None
     


def calculateRanks(query,gallery,g_id,Bayes):
    ranks=np.zeros((len(gallery),len(query)),int)
    ranks2=np.zeros((len(gallery),len(query)),int)
    ranks_probability=np.zeros((len(gallery),len(query)),float)
    column=0
    for q in query:
        d=np.zeros(len(gallery))
        p=np.zeros(len(gallery))
        i=0
        for g in gallery:
            #d[i]=(1/(1+histogram_intersection(q,g)))
            d[i]=histogram_distance(q,g)
            p[i]=Bayes.calculateProbBayes(d[i])
            i+= 1
        sorted_i=np.argsort(-p)
        ranks[:,column] = sorted_i
        ranks2[:,column] = g_id[sorted_i]

        ranks_probability[:,column]= -np.sort(-p)
        column += 1
    return ranks,ranks_probability,ranks2 

def queryExpansion(ranking,gallery,query,K):
    ranks_index,ranks_probability=ranking[0],ranking[1]
    q_expansion=[]
    for i in range(len(query)):
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        q_exp=0 #Query originale
        probability_sum=0
        for j in range(K):
            probability_c=candidates_probability[j]
            c_hist=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
            q_exp=q_exp + probability_c*c_hist
            probability_sum += probability_c 
        q_exp=(q_exp +query[i])/(probability_sum +1)
        q_expansion.append(q_exp)    
    return q_expansion
                
if __name__ == '__main__':
    
    n=1000
    H,ident=hist_train,id_t
    x=H[0:n]
    y=ident[0:n]
    print('START TRAINING')
    
    B=BayesianModel()
    B.train(x,y,100)
    print('TRAINING COMPLETE')
    
    h1,b1=B.hist_d_sameId
    h2,b2=B.hist_d_differentId 
    
    x1=np.zeros_like(h1)
    x2=np.zeros_like(h2)
    for i in range(len(b1)-1):
        x1[i]=(b1[i] + b1[i+1])/2
        x2[i]=(b2[i] + b2[i+1])/2
    
    
    
    pl.plot(x1,h1,label='sameId',color='r')
    pl.plot(x2,h2,label='differentId',color='b')
     
    pl.legend()
    pl.show()  
                
    

    

                
                
                
                
                
                
        