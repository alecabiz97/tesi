# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:14:29 2020

@author: aleca
"""
import os
import glob
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from utils.Lbp import *
from utils.hog import *
from utils.BayesianModel import *
import time
import random
import pickle

def confronto_mAP_k(data,testname,title):
    for data,name in zip(data,testname):

        n_id,q_id,risultati=data
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
        r1_first=vettori_cmc[0][0]
        mAP_first=vettore_mAP[0]    
        x,rank1,mAP=[0],[r1_first],[mAP_first]
        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r           
            x.append(k)
            mAP.append(vettore_mAP[1])
        
        pl.plot(x,np.array(mAP)*100,'-o',label=name)
        pl.ylim(70,100)
    pl.legend()
    pl.grid(True)
    pl.ylabel('mAP(%)')
    pl.xlabel('K')
    pl.title(title)
    pl.show()
 

def confronto_rank1_k(data,testname,title):
    for data,name in zip(data,testname):

        n_id,q_id,risultati=data
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
        r1_first=vettori_cmc[0][0]
        mAP_first=vettore_mAP[0]  
        x,rank1,mAP=[0],[r1_first],[mAP_first]
        print((name,r1_first,mAP_first))
        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r           
            x.append(k)
            r1=vettori_cmc[1][0] #Rank1 dopo la prima iterazione
            rank1.append(r1)
        
        pl.plot(x,np.array(rank1)*100,'-o',label=name)
        pl.ylim(70,100)
    pl.legend()
    pl.grid(True)
    pl.ylabel('rank1(%)')
    pl.xlabel('K')
    pl.title(title)
    pl.show()  

def confronto_mAP_n(data,testname,title):
    pass

def confronto_rank1_n(data,testname,title):
    pass




dirDuke='..\\Risultati test\\Duke300Feedback'
dirMarket='..\\Risultati test\\Market300Feedback'
filesname,data=loadFiles(dirMarket)

testname=['EMR','Rocchio','Feedback Non Pesato']
titleMarket='Market-1501'
titleDuke='DukeMTMC-reId'


confronto_mAP_k(data,testname,titleMarket)
confronto_rank1_k(data,testname,titleMarket)



#


