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
from evaluation import *
from Lbp import *
from hog import *
from BayesianModel import *
import time
import random
import pickle

#Dir='..//Risultati test//Duke_test_complete_withoutRanks.pkl'
Dir='..//Risultati test//Market_test_complete_withoutRanks.pkl'
        
f=open(Dir,'rb')
results=pickle.load(f)
f.close()

n_id,q_id,risultati=results

#Mostra l'andamento di mAP e rank1 all variare delle iterazioni    
for r in risultati:
    k,n,vettori_cmc,vettore_mAP=r
    x=np.arange(0,n+1)
    rank1=[v[0] for v in vettori_cmc]
    print(k)
    pl.plot(x,vettore_mAP,'-o',label='mAP')
    pl.plot(x,rank1,'-o',label='Rank1')
    pl.legend()
    pl.grid(True)
    pl.ylabel('Probability')
    pl.xlabel('Iterazioni')
    
    pl.show()        

#Mostra la cmc per ogni iterazione
for r in risultati:
    k,n,vettori_cmc,vettori_mAP=r
    i=0
    print(k)
    for y in vettori_cmc:
        x1=np.arange(1,len(y)+1)
        pl.plot(x1,y,label='Iterazione: {}'.format(i))
        i += 1
    pl.legend()
    pl.grid(True)
    pl.xlim([1,100])
    pl.ylabel('Probability')
    pl.xlabel('Rank')
    pl.plot()
    pl.show()        
        
        
        
        
        
        
        
        
         
