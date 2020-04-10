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
    
    def __init__(self,K=50):
        self.K=K
        
    def train(self,gallery_train,gallery_labels,query_train,query_labels,M=100):
        gallery_labels=np.array(gallery_labels)
        query_labels=np.array(query_labels)
        
        ranks=np.zeros((len(gallery_train),len(query_train)))
        d_sameId,d_differenId=[],[]
        column=0
        for q,i in zip(query_train,range(len(query_train))):
            d=np.zeros(len(gallery_train))
            j=0
            for h in gallery_train:
                    d[j]=(1/(1+histogram_intersection(q,h)))  #Calcolo distanza
                    j+= 1
            sorted_id=np.argsort(d)
            sorted_labels=gallery_labels[sorted_id]
            
            label_q=query_labels[i]
            
            #Distanze tra le immagini della stessa persona
            for val in d[sorted_labels==label_q]:
                d_sameId.append(val) 
            #Distanze tra le immagini di persone diverse
            for val in d[sorted_labels!=label_q]:
                d_differenId.append(val)  
            
            ranks[:,column] = sorted_labels   
            column += 1
        
        d_sameId,d_differenId=np.array(d_sameId),np.array(d_differenId)
        
        #np.histogram() restituisce una lista che contiene l'istgramma e un vettore con gli estremi dei bins
        self.hist_d_sameId=np.histogram(d_sameId,M) 
        self.hist_d_differenId=np.histogram(d_differenId,M) 
        
        label_found=0
        i=0
        #Controllo quante volte la label-iesima si trova tra le prime K
        for label in query_labels:
            label_found += np.sum(ranks[0:self.K,i]==label)
            i += 1
    
        self.P_ltiEqualsltj=label_found/len(query_labels)
        self.P_ltiNotEqualsltj= 1-self.P_ltiEqualsltj
        
    def calculateP_d(self,distanza):
        h_sameId, bins_sameId = self.hist_d_sameId
        h_diffId, bins_diffId = self.hist_d_differenId
        
        #Calcolo P_d_lqEqualslg
        distanza_index=np.sum(distanza>bins_sameId)-1 #è l'indice dell'intervallo in cui cade distanza
         
        P_d_lqEqualslg = h_sameId[distanza_index]/np.sum(h_sameId)
        
        #Calcolo P_d_lqNotEqualslg
        distanza_index=np.sum(distanza>bins_diffId)-1 #è l'indice dell'intervallo dell'istogramma in cui cade distanza

        P_d_lqNotEqualslg = h_diffId[distanza_index]/np.sum(h_diffId)
        
        P_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)+(P_d_lqNotEqualslg*self.P_ltiNotEqualsltj)

        #Teorema di Bayes
        P_lqEqualslg_d=(P_d_lqEqualslg*self.P_ltiEqualsltj)/P_d
        return P_lqEqualslg_d
     


def test(query,gallery,Bayes):
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
    
#    gallery=[histogram_vector(camA[i]) for i in range(10)]
#    query=[histogram_vector(camB[i]) for i in range(10)]
#    
#    r1,r2=ranking(query,gallery)
    
         

    random_id=np.random.permutation(np.arange(0,len(camA[0:400])))
    length_train=int(0.8*len(random_id))
    train_index=random_id[0:length_train]
    test_index=random_id[length_train::]
    
    gallery_train=[histogram_vector(camA[i]) for i in train_index]
    query_train=[histogram_vector(camB[i]) for i in train_index]  
    gallery_train_labels=[Id_A[i] for i in train_index]
    query_train_labels=[Id_B[i] for i in train_index]
    
    gallery_test=[histogram_vector(camA[i]) for i in test_index]
    query_test=[histogram_vector(camB[i]) for i in test_index]
    y_test=[Id_B[i] for i in test_index]


    
    
    B=BayesianModel()
    B.train(gallery_train,gallery_train_labels,query_train,query_train_labels) 
    

    print('Start test')
    

    r=[]
    for i in range(2):
        ranks=test(query_test,gallery_test,B)
        print('Ranks calcolato')
        query_test=queryExpansion(ranks,gallery_test,query_test,20) 
        print('Nuova query calcolata')

        r.append(ranks[0])
        
                
                
                
                
                
                
                
                
                
                
        