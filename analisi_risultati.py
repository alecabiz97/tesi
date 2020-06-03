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

#Mostra l'andamento di mAP  all variare di k per i diversi metodi
def plot_mAP_functionOfK(directory,testname,title):
    for Dir,name in zip(directory,testname):
        f=open(Dir,'rb')
        results=pickle.load(f)
        f.close()
    
        n_id,q_id,risultati=results
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
        r1_first=vettori_cmc[0][0]
        mAP_first=vettore_mAP[0]    
        x,rank1,mAP=[0],[r1_first],[mAP_first]
        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r           
            x.append(k)
            mAP.append(vettore_mAP[1])
        
        pl.plot(x,np.array(mAP)*100,'-o',label=name)
        pl.ylim(50,100)
    pl.legend()
    pl.grid(True)
    pl.ylabel('mAP(%)')
    pl.xlabel('K')
    pl.title(title)
    pl.show()
 
#Mostra l'andamento di rank1 all variare di k per i diversi metodi
def plot_rank1_functionOfK(directory,testname,title):
    for Dir,name in zip(directory,testname):
        f=open(Dir,'rb')
        results=pickle.load(f)
        f.close()
    
        n_id,q_id,risultati=results
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
        r1_first=vettori_cmc[0][0]
        mAP_first=vettore_mAP[0]    
        x,rank1,mAP=[0],[r1_first],[mAP_first]
        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r           
            x.append(k)
            r1=vettori_cmc[1][0] #Rank1 dopo la prima iterazione
            rank1.append(r1)
        
        pl.plot(x,np.array(rank1)*100,'-o',label=name)
        pl.ylim(50,100)
    pl.legend()
    pl.grid(True)
    pl.ylabel('rank1(%)')
    pl.xlabel('K')
    pl.title(title)
    pl.show()   

#Mostra l'andamento di rank1 all variare di n per i diversi metodi
def plot_rank1_functionOfn(directory,testname,title):
    for Dir,name in zip(directory,testname):
        f=open(Dir,'rb')
        results=pickle.load(f)
        f.close()
    
        n_id,q_id,risultati=results
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]

        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r 
            if ('withFeedback' in Dir and k==55) or ('withoutFeedback' in Dir and k==5):
                x=np.arange(n+1)
                rank1=[v[0] for v in vettori_cmc] #Rank1 dopo la prima iterazione
                pl.plot(x,np.array(rank1)*100,'-o',label=name)
    pl.ylim(70,100)
    pl.legend(loc='lower right')
    pl.grid(True)
    pl.ylabel('rank1(%)')
    pl.xlabel('n')
    pl.title(title)
    pl.show() 

#Mostra l'andamento di mAP all variare di n per i diversi metodi
def plot_mAP_functionOfn(directory,testname,title):
    for Dir,name in zip(directory,testname):
        f=open(Dir,'rb')
        results=pickle.load(f)
        f.close()
    
        n_id,q_id,risultati=results
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]

        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r 
            if ('withFeedback' in Dir and k==55) or ('withoutFeedback' in Dir and k==5):
                x=np.arange(n+1)
                pl.plot(x,np.array(vettore_mAP)*100,'-o',label=name)
    pl.ylim(70,100)
    pl.legend(loc='lower right')
    pl.grid(True)
    pl.ylabel('mAP(%)')
    pl.xlabel('n')
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

def evaluation_forEachIdentity(ranks,id_query):
    resultati_totali=[]
    id_query=np.array(id_query)
    labels=list(set(id_query))
    for q in labels:
        colonne=np.array(id_query==q)
        rank_tmp=ranks[:,colonne]
        rank1=calculateCmcFromRanks(rank_tmp,[q])[0] #Calcolo rank1
        mAP=(calculate_mAP(rank_tmp,[q],len(rank_tmp)))
        if rank1 < 0.9:  
            results=[q]
            results.append(rank1)
            results.append(mAP)
            resultati_totali.append(results)
    return resultati_totali
                  
if __name__ == '__main__':


          
#Duke
#    Dir='..//Risultati test//Duke//Duke_test_complete.pkl'
#    Dir='..//Risultati test//Duke//Duke_results_100Id.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_randomK5.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_randomK10.pkl'

    Dir1='..//Risultati test//Duke//Duke_300pics_k_n_withoutFeedback_AQE.pkl'
    Dir2='..//Risultati test//Duke//Duke_300pics_k_n_withoutFeedback_soglia0,5.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_AQE.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_AQE_Similarity.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_soglia0,5.pkl'    
#    Dir='..//Risultati test//Duke//Duke_test_complete_Similarity.pkl'
    
    #Feedback   
     
    Dir3='..//Risultati test//Duke//Duke_300pics_k_n_withFeedback_Prob.pkl'
    Dir4='..//Risultati test//Duke//Duke_300pics_k_n_withFeedback_Prob1.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'  
#    Dir='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob1_k55.pkl'   
#    Dir='..//Risultati test//Duke//Duke_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'    

    
##############################################
#Market  

    Dir5='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_AQE.pkl'
    Dir6='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_soglia0,5.pkl' 
#    Dir='..//Risultati test//Market//Market_test_complete.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_Similarity.pkl'    
#    Dir='..//Risultati test//Market//Market_test_complete_AQE.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_AQE_Similarity.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_soglia0,5.pkl'    

    #Feedback   

    Dir7='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob.pkl'
    Dir8='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob1.pkl' 
#    Dir='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55.pkl'
    Dir='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'    
#    Dir='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob1_k55.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'
    
#################################################
    
#    f=open(Dir,'rb')
#    results=pickle.load(f)
#    f.close()
#    
#    n_id,q_id,risultati=results
    
#    title='Market-1501'
#    title='DukeMTMC-reID'
        
#    rank1_mAP_functionOfK(risultati,title)
#    print(Dir)
#    rank1_mAP_functionOfn(risultati,title)
    
#    plotCMC_forEachIteration(risultati,title)
    
    directory_duke=[Dir1,Dir2,Dir3,Dir4]
    nuova_cartella='C://Users//aleca//Desktop//Nuova cartella//'
    directory_duke2=[nuova_cartella +d.split('//')[-1] for d in directory_duke]
    directory_market=[Dir5,Dir6,Dir7,Dir8]
    testname=['AQE','BQE','Feedback pesato','Feedback non pesato']
    title_duke='DukeMTMC-reID'
    title_market='Market-1501'

#    plot_mAP_functionOfK(directory_market,testname,title)
#    plot_rank1_functionOfK(directory_duke,testname,title_duke)
    
    plot_rank1_functionOfn(directory_duke2,testname,title_duke)
    plot_mAP_functionOfn(directory_duke2,testname,title_duke)

    
    

  
            



  



