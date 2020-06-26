# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:27:14 2020

@author: aleca
"""

import os
import pandas as pd
import numpy as np
from importData import *
from evaluation import *
import openpyxl
import pickle
from openpyxl.utils.dataframe import dataframe_to_rows

DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'
DirDataset=DirMarket
Dir1='C:\\Users\\aleca\\Desktop\\Nuova cartella\\Ranksi-DukeFromMarket'

n,X=loadFiles("C:\\Users\\aleca\\Desktop\\Nuova cartella\\Nuova cartella")
testData,queryData,trainingData=loadCNN(DirMarket)

test_cams, test_feature, test_id, test_desc = testData
query_cams, query_feature, query_id, query_desc = queryData
train_cams, train_feature, train_id, train_desc = trainingData

gallery_id,gallery_cams=np.array(test_id),np.array(test_cams)

rows_baseline=[]
rows=[]
for results,filename in zip(X,n):
    newFilename=filename.split('-')[-1]
    print(newFilename)
    n_id,query_ids,risultati=results
    X2=[n_id,query_ids]
    x2=[]
    for r in risultati:
        k,n,allranks_index=r
        xk=[k,n]
        vettori_cmc,vettore_mAP=[],[]
        for ranks_index in allranks_index:
            #Creo ranks label
            ranks_label=gallery_id[ranks_index]
            
            cmc=calculateCmcFromRanks(ranks_index,ranks_label,query_ids,gallery_cams,query_cams,topk=100)
            mAP=calculate_mAP(ranks_index,ranks_label,query_ids,gallery_cams,query_cams)
            print((cmc[0],mAP))
            vettori_cmc.append(cmc)
            vettore_mAP.append(mAP)
        xk.append(vettori_cmc)
        xk.append(vettore_mAP)
        x2.append(xk)
    X2.append(x2)
    
#    f=open(newFilename,'wb')
#    pickle.dump(X2,f)
#    f.close()
        
        
        
        
        
        
        
