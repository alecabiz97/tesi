# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:28:01 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
import time
import random
from scipy.spatial import distance



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
        hist_sameId,bins_sameId=np.histogram(d_sameId,M)
        hist_differentId,bins_diffId=np.histogram(d_differentId,M)
        
        self.hist_d_sameId=[hist_sameId,bins_sameId]
        self.hist_d_differentId=[hist_differentId,bins_diffId]
        
        self.d_sameId=d_sameId
        self.d_differentId=d_differentId
                
        
    def calculateProbBayes(self,distanza):
        #h_sameId: istogramma distanze sameId 
        #h_diffId: istogramma distanze diffId 

        #bins_sameId: valori estremi bins istogramma sameId
        #bins_diffId: valori estremi bins istogramma diffId
        h_sameId, bins_sameId = self.hist_d_sameId
        h_diffId, bins_diffId = self.hist_d_differentId
        
        #Calcolo P_d_sameId
        distanza_index_sameId=np.sum(distanza>bins_sameId)-1 #è l'indice dell'intervallo in cui cade distanza
        if distanza_index_sameId<0:
            distanza_index_sameId += 1
            P_d_sameId = h_sameId[distanza_index_sameId]/np.sum(h_sameId)
        if distanza_index_sameId == len(h_sameId):
            P_d_sameId=0
        else:
            P_d_sameId = h_sameId[distanza_index_sameId]/np.sum(h_sameId)
            
        #Calcolo P_d_diffId
        distanza_index_diffId=np.sum(distanza>bins_diffId)-1 #è l'indice dell'intervallo dell'istogramma in cui cade distanza
        if distanza_index_diffId<0:
            P_d_diffId = 0
        elif distanza_index_diffId == len(h_diffId):
            distanza_index_diffId -= 1
            P_d_diffId = h_diffId[distanza_index_diffId]/np.sum(h_diffId)
        else:
            P_d_diffId = h_diffId[distanza_index_diffId]/np.sum(h_diffId)
                
        #Calcolo P_d    
        P_d=(P_d_sameId*self.P_ltiEqualsltj)+(P_d_diffId*self.P_ltiNotEqualsltj)
        if P_d == 0:
            #Se P_d== 0 (d circa 0.2) quindi prendo il valore di probabilità adiacente
            P_d_sameId = h_sameId[distanza_index_sameId-1]/np.sum(h_sameId)
            P_d=(P_d_sameId*self.P_ltiEqualsltj)+(P_d_diffId*self.P_ltiNotEqualsltj)

            
        #Teorema di Bayes
        P_sameId_d=(P_d_sameId*self.P_ltiEqualsltj)/P_d
        return P_sameId_d
    
    def plotTrainingHistogram(self,norm=False):
        h1,b1=self.hist_d_sameId
        h2,b2=self.hist_d_differentId
       
        x1=np.zeros_like(h1,float)
        x2=np.zeros_like(h2,float)
        for i in range(len(b1)-1):
            x1[i]=(b1[i] + b1[i+1])/2
            x2[i]=(b2[i] + b2[i+1])/2
        
        width_binsSame=(max(b1)-min(b1))/100
        width_binsDiff=(max(b2)-min(b2))/100
        
        if norm==True:
           h1=(h1/np.sum(h1))/width_binsSame
           h2=(h2/np.sum(h2))/width_binsDiff
        
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
    ranks_index=np.zeros((len(gallery),len(query)),int)
    ranks_label=np.zeros((len(gallery),len(query)),int)
    ranks_probability=np.zeros((len(gallery),len(query)),float)
    column=0
    for q in query:
        p=np.zeros(len(gallery))
        i=0
        for g in gallery:
            d=histogram_distance(q,g)
            p[i]=Bayes.calculateProbBayes(d)
            i+= 1
        sorted_i=np.argsort(-p)
        ranks_index[:,column] = sorted_i
        ranks_label[:,column] = [g_id[i] for i in sorted_i]
        ranks_probability[:,column]= -np.sort(-p)
        column += 1
    return ranks_index,ranks_probability,ranks_label



def calculateRanks_Similarity(query,gallery,g_id,Bayes):
    ranks_index=np.zeros((len(gallery),len(query)),int)
    ranks_label=np.zeros((len(gallery),len(query)),int)
    ranks_similarity=np.zeros((len(gallery),len(query)),float)
    column=0
    for q in query:
        s=np.zeros(len(gallery))
        i=0
        for g in gallery:
            s[i]=1/(1+histogram_distance(q,g))
            i+= 1
        sorted_i=np.argsort(-s)
        ranks_index[:,column] = sorted_i
        ranks_label[:,column] = [g_id[i] for i in sorted_i]
        ranks_similarity[:,column]= -np.sort(-s)
        column += 1
    return ranks_index,ranks_similarity,ranks_label  

def calculateRanks_Distance(query,gallery,g_id,Bayes):
    ranks_index=np.zeros((len(gallery),len(query)),int)
    ranks_label=np.zeros((len(gallery),len(query)),int)
    ranks_similarity=np.zeros((len(gallery),len(query)),float)
    column=0
    for q in query:
        d=np.zeros(len(gallery))
        i=0
        for g in gallery:
            d[i]=histogram_distance(q,g)
            i+= 1
        #sorted_i=np.argsort(-s)
        #ranks_index[:,column] = sorted_i
        #ranks_label[:,column] = [g_id[i] for i in sorted_i]
        ranks_similarity[:,column]= d
        column += 1
    return np.transpose(ranks_similarity)  



    
                
if __name__ == '__main__':
    
    
    DirMarket = '..\\FeatureCNN\\Market-1501'
    DirDuke = '..\\FeatureCNN\\DukeMTMC'
    
    #Feature CNN
    testData,queryData,trainingData=loadCNN(DirMarket)
    
    #istogrammi RGB
    #testData,queryData,trainingData=loadMarket_1501(feature=True)
    
    test_cams, test_feature, test_id, test_desc = testData
    query_cams, query_feature, query_id, query_desc = queryData
    train_cams, train_feature, train_id, train_desc = trainingData
    
      
    print('START')
    
    #Load BayesianModel gia addestrato
    Bayes=loadFile('..\\Bayes_Market_trained.pkl')
    Bayes=loadFile('..\\Bayes_Duke_trained.pkl')
    print('TRAINING COMPLETE')
    #Bayes.plotTrainingHistogram(True)
    d=np.arange(0,3,0.005)
    s=[1/(1+i) for i in d]
    p=[Bayes.calculateProbBayes(i) for i in d]
    X=np.array([d,s,p])
    
    pl.plot(d,s,label='Similarità')
    pl.plot(d,p,label='Probabilità sameId')
    pl.title('Market')
    pl.legend()
    pl.grid()
    pl.show()
    
#    gallery,g_id=test_feature,test_id
#    query,q_id = query_feature[0:400], query_id[0:400]
#    
#    n,k=0,5
#    q1,q2=query,query
#    for i in range(n+1):
#        r1,r2,r3=calculateRanks(q1,gallery,g_id,Bayes)
#
#        #Probabilità
#        mAP=calculate_mAP(r3,q_id,r3.shape[0])
#        rank1=calculateCmcFromRanks(r3,q_id)[0]
#        print(('Prob',mAP,rank1))
#        
#        rr1,rr2,rr3=calculateRanks_Similarity(q2,gallery,g_id,Bayes)
#
#        #Similarità
#        mAP=calculate_mAP(rr3,q_id,rr3.shape[0])
#        rank1=calculateCmcFromRanks(rr3,q_id)[0]
#        print(('Sim',mAP,rank1))
#    
#        q1=queryExpansion(r1,r2,gallery,query,k,AQE=False,soglia=0)
#        q2=queryExpansion(rr1,rr2,gallery,query,k,AQE=False,soglia=0)
#    
#        print('\n')
#    
     
#    Dir9='..//FeatureCrossDataset//dukeMTMCfrommarket1501_gallery_descriptors.pkl'
#    Dir10='..//FeatureCrossDataset//dukeMTMCfrommarket1501_query_descriptors.pkl'
#    Dir11='..//FeatureCrossDataset//market1501fromdukeMTMC_gallery_descriptors.pkl'
#    Dir12='..//FeatureCrossDataset//market1501fromdukeMTMC_query_descriptors.pkl'
#    
#    f=open(Dir9,'rb')
#    dukeFromMarket_gallery=pickle.load(f)
#    f.close()
#    
#    f=open(Dir10,'rb')
#    dukeFromMarket_query=pickle.load(f)
#    f.close()
#    
#    f=open(Dir11,'rb')
#    marketFromDuke_gallery=pickle.load(f)
#    f.close()
#    
#    f=open(Dir12,'rb')
#    marketFromDuke_query=pickle.load(f)
#    f.close() 


        #Cross dataset
#    gallery,g_id=dukeFromMarket_gallery,test_id
#    query,q_id = dukeFromMarket_query[0:100], query_id[0:100]
#

            
                
             
    
                
                
        