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
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from utils.Lbp import *
from utils.hog import *
from BayesianModel import *
import time
import random
import pickle

#Mostra l'andamento di mAP alla prima iterazione all variare di k per i diversi metodi
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
 
#Mostra l'andamento di rank1 alla prima iterazione all variare di k per i diversi metodi
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

#Mostra l'andamento di rank10 alla prima iterazione all variare di k per i diversi metodi
def plot_rank10_functionOfK(directory,testname,title):
    for Dir,name in zip(directory,testname):
        f=open(Dir,'rb')
        results=pickle.load(f)
        f.close()
    
        n_id,q_id,risultati=results
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]
        r10_first=vettori_cmc[0][9]
        x,rank10=[0],[r10_first]
        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r           
            x.append(k)
            r10=vettori_cmc[1][9] #Rank10 dopo la prima iterazione
            rank10.append(r10)
        
        pl.plot(x,np.array(rank10)*100,'-o',label=name)
        pl.ylim(50,100)
    pl.legend()
    pl.grid(True)
    pl.ylabel('rank10(%)')
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
            if ('Feedback' in Dir and k==55) or (('AQE' in Dir or 'BQE' in Dir) and k==5):
                x=np.arange(n+1)
                rank1=[v[0] for v in vettori_cmc] #Rank1 dopo la prima iterazione
                pl.plot(x,np.array(rank1)*100,'-o',label=name)
    pl.ylim(50,100)
    pl.legend(loc='lower right')
    pl.grid(True)
    pl.ylabel('rank1(%)')
    pl.xlabel('n')
    pl.title(title)
    pl.show() 
    
#Mostra l'andamento di rank10 all variare di n per i diversi metodi
def plot_rank10_functionOfn(directory,testname,title):
    for Dir,name in zip(directory,testname):
        f=open(Dir,'rb')
        results=pickle.load(f)
        f.close()
    
        n_id,q_id,risultati=results
        
        k,n_iteration,vettori_cmc,vettore_mAP=risultati[0]

        for r in risultati: 
            k,n,vettori_cmc,vettore_mAP=r 
            if ('Feedback' in Dir and k==55) or (('AQE' in Dir or 'BQE' in Dir) and k==5):
                x=np.arange(n+1)
                rank10=[v[9] for v in vettori_cmc] #Rank1 dopo la prima iterazione
                pl.plot(x,np.array(rank10)*100,'-o',label=name)
    pl.ylim(50,100)
    pl.legend(loc='lower right')
    pl.grid(True)
    pl.ylabel('rank10(%)')
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
            if ('Feedback' in Dir and k==55) or (('AQE' in Dir or 'BQE' in Dir) and k==5):
                x=np.arange(n+1)
                pl.plot(x,np.array(vettore_mAP)*100,'-o',label=name)
    pl.ylim(50,100)
    pl.legend(loc='lower right')
    pl.grid(True)
    pl.ylabel('mAP(%)')
    pl.xlabel('n')
    pl.title(title)
    pl.show()     
    
            
                  
if __name__ == '__main__':



          
#Duke
#    Dir='..//Risultati test//Duke//Duke_test_complete.pkl'
#    Dir='..//Risultati test//Duke//Duke_results_100Id.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_randomK5.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_randomK10.pkl'

    Dir1='..//Risultati test//Duke300//Duke_300pics_k_n_AQE.pkl'
    Dir2='..//Risultati test//Duke300//Duke_300pics_k_n_BQE.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_AQE.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_AQE_Similarity.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_soglia0,5.pkl'    
#    Dir='..//Risultati test//Duke//Duke_test_complete_Similarity.pkl'
    
    #Feedback   
     
    Dir3='..//Risultati test//Duke300//Duke_300pics_k_n_FeedbackPesato.pkl'
    Dir4='..//Risultati test//Duke300//Duke_300pics_k_n_Feedback_NonPesato.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55.pkl'
#    Dir='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'  
#    Dir='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob1_k55.pkl'   
#    Dir='..//Risultati test//Duke//Duke_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'    

    
##############################################
#Market  

    Dir5='..//Risultati test//Market300//Market_300pics_k_n_AQE.pkl'
    Dir6='..//Risultati test//Market300//Market_300pics_k_n_BQE.pkl' 
#    Dir='..//Risultati test//Market//Market_test_complete.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_Similarity.pkl'    
#    Dir='..//Risultati test//Market//Market_test_complete_AQE.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_AQE_Similarity.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_soglia0,5.pkl'    

    #Feedback   

    Dir7='..//Risultati test//Market300//Market_300pics_k_n_FeedbackPesato.pkl'
    Dir8='..//Risultati test//Market300//Market_300pics_k_n_Feedback_NonPesato.pkl' 
#    Dir='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'    
#    Dir='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob1_k55.pkl'
#    Dir='..//Risultati test//Market//Market_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'
    
#################################################
#    dir_resultTest="..\\Risultati test\\"
#    filenames=['Duke_Rocchio.pkl','Market_Rocchio.pkl','DukeFromMarket_Rocchio.pkl','MarketFromDuke_Rocchio.pkl']
#    for filename in filenames:
#        f=open(dir_resultTest+filename,'rb')
#        n_id,q_id,risultati=pickle.load(f)
#        
#        for r in risultati:
#            k,n,v_cmc,v_mAP=r
#            r1=[v[0] for v in v_cmc]
#            r10=[v[9] for v in v_cmc]
#        
#        print(filename.split('.')[0])
#        for i in range(4):
#            print('Round {} mAP: {:.2f}  rank1: {:.2f}  rank10: {:.2f}'.format(i,v_mAP[i]*100,r1[i]*100,r10[i]*100))
#        print('\n')
    
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
    
    Dir9="..\\Risultati test\\Duke300_Rocchio.pkl"
    Dir10="..\\Risultati test\\Market300_Rocchio.pkl"
    
    directory_duke=[Dir1,Dir2,Dir3,Dir4,Dir9]
    
    directory_market=[Dir5,Dir6,Dir7,Dir8,Dir10]
    
    testname=['AQE','BQE','Feedback pesato','Feedback non pesato','Rocchio']
    title_duke='DukeMTMC-reID'
    title_market='Market-1501'

    plot_mAP_functionOfK(directory_duke,testname,title_duke)
#    plot_rank1_functionOfK(directory_duke,testname,title_duke)
    
#    plot_mAP_functionOfn(directory_market,testname,title_market)
#    plot_rank1_functionOfn(directory_duke,testname,title_duke)

#    plot_rank10_functionOfK(directory_duke,testname,title_duke)
#    
#    plot_rank10_functionOfn(directory_duke,testname,title_duke)

    

  
            



  



