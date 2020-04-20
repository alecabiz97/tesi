# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:14:33 2020

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



def test_Cmc_Cnn():
    DirDuke = '..\\FeatureCNN\\DukeMTMC'
    
    testDuke,queryDuke,trainDuke=loadCNN(DirDuke)
    gallery_cams, gallery_cnn, gallery_id, gallery_desc = testDuke
    query_cams, query_cnn, query_id, query_desc = queryDuke
    train_cams, train_cnn, train_id, train_desc = trainDuke
    
    set_of_probes=query_cnn[0:20]
    id_probes=query_id[0:20]
    
    gallery=gallery_cnn
    id_gallery=gallery_id

    print('START')
    
    cmc_vector= cmc(set_of_probes, id_probes, gallery, id_gallery)
    plot_CMC(cmc_vector)  
    #print(sorted(positions))



if __name__ == '__main__':
    
      
    DirMarket = '..\\FeatureCNN\\Market-1501'
    DirDuke = '..\\FeatureCNN\\DukeMTMC'
    
    #testDuke,queryDuke,trainDuke=loadCNN(DirDuke)
    gallery_cams, gallery_cnn, gallery_id, gallery_desc = testDuke
    query_cams, query_cnn, query_id, query_desc = queryDuke
    train_cams, train_cnn, train_id, train_desc = trainDuke
    
    print('Feature importate')
    print('Start')
    n=8000
    
    B=BayesianModel()
    B.train(train_cnn[0::],train_id[0::])
    print('TRAINING COMPLETE')
    
    
    print('START TEST')
    
    t=10000
    test,t_id=gallery_cnn[0::],gallery_id[0::]
    query,q_id=query_cnn[0:20],query_id[0:20]
    
    t_id=np.array(t_id)
    
    print(q_id)
    
    
    r=[]
    for i in range(3):
        ranks=calculateRanks(query,test,t_id,B)
        print('Ranks calcolato')
        
        
        query=queryExpansion(ranks,test,query,5) 
        print('Nuova query calcolata')
    
        r.append(ranks[2])
        
    r1=r[0]
    r2=r[1]
    r3=r[2]
    
    for i in range(len(q_id)):
        q=q_id[i]
        print((q,np.sum(q==r1[0:6,i]),np.sum(q==r2[0:6,i]),np.sum(q==r2[0:6,i])))
        
    
    
