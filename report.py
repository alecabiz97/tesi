# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:11:37 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from histogram import *
from cmc import *
from BayesianModel import *
import time
import random

def distanceHistogram(gallery,id_gallery,M):
    
    hist_sameId,hist_differentId=[],[]
        
    labels=np.random.permutation(list(set(id_gallery)))
    
    for y in labels:
        d_sameId,d_differentId=[],[]
        for i in range(len(gallery)):
            for j in range(len(gallery)):
                if j > i and (id_gallery[i] == y or id_gallery[j] == y ):
                    d=histogram_distance(gallery[i],gallery[j])
                    #d=histogram_intersection(train[i],train[j])
                    if id_gallery[i] == id_gallery[j] and id_gallery[j] == y :
                        d_sameId.append(d)
                    else:
                        d_differentId.append(d)
                       
        d_sameId,d_differentId=np.array(d_sameId),np.array(d_differentId)
        
        
        h1,b1=np.histogram(d_sameId,M)
        h2,b2=np.histogram(d_differentId,M)
        
        #np.histogram() restituisce una lista che contiene l'istogramma e un vettore con gli estremi degli intervalli
        hist_sameId.append([h1,b1,y])
        hist_differentId.append([h2,b2,y])
        
    return hist_sameId,hist_differentId,
        
        
if __name__ == '__main__':
    
    #Load Market
    DirMarket = '..\\FeatureCNN\\Market-1501'    
#    testMarket,queryMarket,trainMarket=loadCNN(DirMarket)
    
    train_cams_Market, train_cnn_Market, train_id_Market, train_desc_Market = trainMarket
    
    print('Feature importate')
    
    
    
    id_set=list(set(train_id_Market))
    X=np.zeros((len(id_set),2),int)
    j=0
    for i in id_set:
        X[j]=[i,train_id_Market.count(i)]
        j += 1
    
    sorted_i=np.argsort(-X[:,1])
    
    
    
    y=X[sorted_i][0:10][:,0]
    
    train,train_id=[],[]
    for i in range(len(train_id_Market)):
        if train_id_Market[i] in y:
            train.append(train_cnn_Market[i])
            train_id.append(train_id_Market[i])
        
    #train,train_id=train_cnn_Market[0:100],train_id_Market[0:100]
    print('START')
    
#    hist_sameId,hist_differentId=distanceHistogram(train,train_id,100)
        
    
    i=1
    
    print(len(hist_sameId))
    for h_same,h_diff in zip(hist_sameId,hist_differentId):
        h1,b1,y1=h_same
        h2,b2,y2=h_diff
        x1=np.zeros_like(h1,float)
        x2=np.zeros_like(h2,float)
        for i in range(len(b1)-1):
            x1[i]=(b1[i] + b1[i+1])/2
            x2[i]=(b2[i] + b2[i+1])/2
        
        width_binsSame=(max(b1)-min(b1))/100
        width_binsDiff=(max(b2)-min(b2))/100
        
        h1_n=(h1/np.sum(h1))/width_binsSame
        h2_n=(h2/np.sum(h2))/width_binsDiff
        
#        h1_n=h1/np.sum(h1)
#        h2_n=h2/np.sum(h2)
#        pl.subplot(5,2,i)
        pl.bar(x1,h1_n,width_binsSame,label='sameId',color='r')
        pl.bar(x2,h2_n,width_binsDiff,label='differentId',color='b')
#        i += 1
        
        pl.legend()
        pl.xlabel('Distance')
        pl.title('Identità: {}-{}-{}'.format(y1,np.sum(h1),np.sum(h2)))
        pl.show()
