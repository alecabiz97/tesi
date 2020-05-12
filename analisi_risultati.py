#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:49:32 2020

@author: mac
"""

import os
import glob
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


#Mostra l'andamento di mAP e rank1 all variare di k
def Rank1_mAP_functionOfK(risultati,title=''):
    if len(risultati[0]) == 4:
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
    else:
        print('ERROR')
        
    r1_first=vettori_cmc[0][0]
    mAP_first=vettore_mAP[0]    
    x,rank1,mAP=[0],[r1_first],[mAP_first]
    for r in risultati: 
        k,n,vettori_cmc,vettore_mAP=r           
        x.append(k)
        r1=vettori_cmc[1][0] #Rank1 dopo la prima iterazione
        rank1.append(r1)
        mAP.append(vettore_mAP[1])
    
    pl.plot(x,mAP,'-o',label='mAP')
    pl.plot(x,rank1,'-o',label='Rank1')
    pl.legend()
    pl.grid(True)
    pl.ylabel('Probability')
    pl.xlabel('K')
    pl.title(title)
    pl.show()
 
#Mostra l'andamento di mAP e rank1 all variare di n    
def Rank1_mAP_functionOfn(risultati,title=''):
    if len(risultati[0]) == 4:
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
    else:
        print('ERROR')
        
    x=np.arange(n_iteration+1)
    for r in risultati:
        k,n,vettori_cmc,vettore_mAP=r           
        print(k)
        rank1=[v[0] for v in vettori_cmc]
        pl.plot(x,vettore_mAP,'-o',label='mAP')
        pl.plot(x,rank1,'-o',label='Rank1')
        pl.legend()
        pl.grid(True)
        pl.ylabel('Probability')
        pl.xlabel('Iteration(n)')
        pl.title(title)
        pl.show()

#Mostra la cmc per ogni iterazione
def plotCMC_forEachIteration(risultati,title=''):
    for r in risultati:
        if len(r) == 4:
            k,n,vettori_cmc,vettore_mAP=r
        else:
            print('ERROR')
            
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
        pl.title(title)
        pl.plot()
        pl.show()     
        
def evaluation_forEachIdentity(all_ranks,id_query):
    resultati_totali=[]
    id_query=np.array(id_query)
    labels=list(set(id_query))
    for q in labels:
        results=[q]
        rank1,mAP=[],[]
        for rank in all_ranks:
            colonne=np.array(id_query==q)
            rank_tmp=rank[:,colonne]
            rank1.append(calculateCmcFromRanks(rank_tmp,[q])[0]) #Calcolo rank1
            mAP.append(calculate_mAP(rank_tmp,[q],len(rank_tmp)))
        results.append(rank1)
        results.append(mAP)
        resultati_totali.append(results)
    return resultati_totali
        
        
if __name__ == '__main__':
    
#    Dir='..//Risultati test//Duke_results_100Id_ranks.pkl'
#    Dir='..//Risultati test//Duke_results_100Id_withoutRanks.pkl'
#    Dir='..//Risultati test//Duke_test_complete_ranks.pkl'
    Dir='..//Risultati test//Duke_test_complete_withoutRanks.pkl'
#    Dir='..//Risultati test//Duke_test_complete_randomK5_ranks.pkl'
#    Dir='..//Risultati test//Duke_test_complete_randomK10_ranks.pkl'
#    Dir='..//Risultati test//Duke_test_complete_Similarity_ranks.pkl'

    
#    Dir='..//Risultati test//Market_results_100Id_ranks.pkl'
#    Dir='..//Risultati test//Market_results_100Id_withoutRanks.pkl'
#    Dir='..//Risultati test//Market_test_complete_ranks.pkl'
#    Dir='..//Risultati test//Market_test_complete_withoutRanks.pkl'
#    Dir='..//Risultati test//Market_test_complete_randomK5_ranks.pkl'
#    Dir='..//Risultati test//Market_test_complete_randomK10_ranks.pkl'
#    Dir='..//Risultati test//Market_test_complete_Similarity_ranks.pkl'


    f=open(Dir,'rb')
    results=pickle.load(f)
    f.close()
    

    n_id,q_id,risultati=results
    
    #title='Market-1501'
    title='DukeMTMC-reID '
    
    Rank1_mAP_functionOfK(risultati,title)
    Rank1_mAP_functionOfn(risultati,title)
    
    plotCMC_forEachIteration(risultati,title)
  
            



  









