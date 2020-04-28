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

def calculateDistance(gallery,id_gallery):
    
    distance=[]
        
    labels=sorted(list(set(id_gallery)))
    
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
        
        distance.append([y,d_sameId,d_differentId])
        
    return distance


##istogrammiDistanze() calcola i due istogrammi separatamente
def istogrammiDistanze(ds,dd,M=100):
   h1,b1=np.histogram(ds,M)
   h2,b2=np.histogram(dd,M)
    
   x1=np.zeros_like(h1,float)
   x2=np.zeros_like(h2,float)
   for i in range(len(b1)-1):
       x1[i]=(b1[i] + b1[i+1])/2
       x2[i]=(b2[i] + b2[i+1])/2

   width_binsSame=(max(b1)-min(b1))/100
   width_binsDiff=(max(b2)-min(b2))/100

   h1_n=(h1/np.sum(h1))/width_binsSame
   h2_n=(h2/np.sum(h2))/width_binsDiff
   
   pl.bar(x1,h1_n,width_binsSame,label='sameId',color='r')
   pl.bar(x2,h2_n,width_binsDiff,label='differentId',color='b')
    
   pl.legend()
   pl.xlabel('Distance')
   pl.show()     
   
   
#istogrammiDistanze2() calcola i due istogrammi utilizzando lo stesso riferimento sul asse x
def istogrammiDistanze2(ds,dd,M=100):
    d=np.concatenate([ds,dd])
    h,b=np.histogram(d,M)
    
    x=np.zeros_like(h,float)
    for i in range(len(b)-1):
        x[i]=(b[i] + b[i+1])/2
    
    h1,h2=np.zeros_like(h),np.zeros_like(h)
    for d in ds:
        distanza_index=np.sum(d>b)-1
        h1[distanza_index] += 1
    
    for d in dd:
        distanza_index=np.sum(d>b)-1
        h2[distanza_index] += 1

    
    width_bins=(max(b)-min(b))/100
    
    h1_n=(h1/np.sum(h1))/width_bins
    h2_n=(h2/np.sum(h2))/width_bins
    
    
    pl.bar(x,h1_n,width_bins,label='sameId',color='r')
    pl.bar(x,h2_n,width_bins,label='differentId',color='b')
    
    pl.legend()
    pl.xlabel('Distance')
    pl.show()
    

    
        
if __name__ == '__main__':
    
 
    #Load Market
    DirMarket = '..\\FeatureCNN\\Market-1501' 
    
    testData,queryData,trainingData=loadCNN(DirMarket)
    test_cams, test_feature, test_id, test_desc = testData
    query_cams, query_feature, query_id, query_desc = queryData
    train_cams, train_feature, train_id, train_desc = trainingData
    
    print('Feature importate')
    
    #Seleziono le prime 100 identità con più immagini
    id_set=list(set(train_id))
    X=np.zeros((len(id_set),2),int)
    j=0
    for i in id_set:
        X[j]=[i,train_id.count(i)]
        j += 1
    sorted_i=np.argsort(-X[:,1])
    y=X[sorted_i][0:100][:,0]
    
    training,training_id=[],[]
    for i in range(len(train_id)):
        if train_id[i] in y:
            training.append(train_feature[i])
            training_id.append(train_id[i])
            
        
    print('START')
    
    distance=calculateDistance(training,training_id)
        
    M=100
    
#    prove=[]
#    print(len(distance))
#    for x in distance:
#        y,ds,dd=x
#        print('{}'.format(y))
#        istogrammiDistanze2(ds,dd,M)
        
    id_prove=[820,148,22]
    prove=[]
    for i in range(len(distance)):
        if distance[i][0] in id_prove:
            prove.append(distance[i])
    
    for x in prove:
        y,ds,dd=x

        h1,b1=np.histogram(ds,M)
        h2,b2=np.histogram(dd,M)
    
        x1=np.zeros_like(h1,float)
        x2=np.zeros_like(h2,float)
        for i in range(len(b1)-1):
           x1[i]=(b1[i] + b1[i+1])/2
           x2[i]=(b2[i] + b2[i+1])/2
    
        width_binsSame=(max(b1)-min(b1))/100
        width_binsDiff=(max(b2)-min(b2))/100
    
        h1_n=(h1/np.sum(h1))/width_binsSame
        h2_n=(h2/np.sum(h2))/width_binsDiff
       
        pl.bar(x1,h1_n,width_binsSame,label='Distance between images of Id {}'.format(y),color='r')
        pl.bar(x2,h2_n,width_binsDiff,label='Distance between images different Id',color='b')
    
        pl.legend()
        pl.xlabel('Distance')
        pl.title('IDENTITY: {}'.format(y))
        pl.show()
    
    #Load BayesianModel gia addestrato
    B_Market=loadFile('..\\B_Market_trained')
    d_same,d_different=B_Market.d_sameId,B_Market.d_differentId
    
    Ds,Dd,Y=[],[],[]
    print((len(d_same),len(d_different)))
    for x in prove:
        y,ds,dd=x    
        Ds.append(ds)
        Dd.append(dd)
        Y.append(str(y))
        
    Ds.append(d_same)
    Dd.append(d_different)
    Y.append('Media')
    pl.boxplot(Dd,labels=Y)
    pl.title('Boxplot of distances between different Id')
    pl.ylabel('Distance')
    pl.xlabel('ID')
    pl.plot()
#    pl.boxplot(Dd,labels=Y)
#    pl.plot()
               
    
    
