# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:08:24 2020

@author: AleCabiz
"""
import os
import glob
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
import random
import time



#Cmc riceve direttamente i vettori delle feature
def cmc1(hist_probes, id_probes, hist_gallery, id_gallery):
    
    #Nelle colonne di all_ranks ci saranno gli indici della gallery ordinati in base alla
    #similarità con la probe
    all_ranks=np.zeros((len(id_gallery),len(id_probes)))
    
    
    K=np.zeros((len(id_probes),1))
    i=0
    for h_probe in hist_probes:
        
        #k è un array con i coeficienti di similarità tra la probe e la gallery
        k=[histogram_intersection(h_probe,h_gallery) for h_gallery in hist_gallery]
        k=np.array(k)
        
        #ordino gli indici in base al k maggiore
        sorted_id=np.argsort(-k)
        #considero gli indici in id_gallery possono non essere continui
        sorted_id_gallery=[id_gallery[i] for i in sorted_id]
        #sorted_id_gallery=id_gallery[sorted_id] #questa non funzionava perche id_gallery è una lista
        
        all_ranks[:,i]=sorted_id_gallery
        i += 1
        
    rank=np.zeros((len(id_gallery),1))
    position_id_matching=[]
    i=0
    for p_id in id_probes:
        position=1
        p_found=False
        rank_tmp=all_ranks[:,i]
        while(p_found == False) and (position-1)<len(rank_tmp):
            if p_id == rank_tmp[position-1]:
                p_found = True
            else:
                p_found = False
                position += 1
        rank[position-1] += 1
        position_id_matching.append(position)
        i +=1
    position_id_matching=np.array(position_id_matching)
    cmc=np.cumsum(rank)/len(id_probes)
    return np.array(cmc),position_id_matching
  
#        val_cmc=np.sum(all_ranks==id_probes,1).cumsum()
#    return val_cmc/len(id_probes)

#Plot CMC
def plot_CMC(cmc_vector):
    x=np.arange(len(cmc_vector))+1
    pl.plot(x,cmc_vector)
    pl.title('Cumulative Match Characteristic')
    pl.ylabel('Probability of Identification')
    pl.xlabel('Rank')
    pl.grid(True)
    pl.show()  

#Prende in ingresso i ranks e calcola la cmc.Filtra utilizzando le camere     
def calculateCmcFromRanks(ranks_index,ranks_label,id_probes,gallery_cams,probes_cams,topk=100):
    rank=np.zeros((len(ranks_label),1))
    gallery_cams=np.array(gallery_cams)
    i=0
    for p_id,p_cam in zip(id_probes,probes_cams):
        sorted_cams=gallery_cams[ranks_index[:,i]]
        j=0
        p_found=False
        if len(ranks_label.shape) == 1:
            rank_tmp=np.array([ranks_label[j] for j in range(len(ranks_label)) if sorted_cams[j] != p_cam])
        else:
            rank_tmp=np.array([ranks_label[:,i][j] for j in range(len(ranks_label)) if sorted_cams[j] != p_cam])
        while p_found == False and j<len(rank_tmp):
            if p_id == rank_tmp[j]:
                p_found = True
            else:
                p_found = False
                j += 1
        rank[j] += 1
        i +=1
    cmc=np.cumsum(rank)/len(id_probes)
    return np.array(cmc[0:topk]) 

#Prende in ingresso i ranks e calcola mAP.Filtra utilizzando le camere
def calculate_mAP(ranks_index,ranks_label,query_ids,gallery_cams,query_cams):
    AP=[]
    gallery_cams=np.array(gallery_cams)
    i=0
    for q_id,q_cam in zip(query_ids,query_cams):
        #Filtro con le camere
        sorted_cams=gallery_cams[ranks_index[:,i]]
        if len(ranks_label.shape) == 1:
            rank_tmp=np.array([ranks_label[j] for j in range(len(ranks_label)) if sorted_cams[j] != q_cam])
        else:
            rank_tmp=np.array([ranks_label[:,i][j] for j in range(len(ranks_label)) if sorted_cams[j] != q_cam])
        #Calcolo il numero di istanze n di q in gallery
        
        n=np.sum(rank_tmp==q_id)
        ap=np.cumsum(rank_tmp==q_id)*(rank_tmp==q_id) 
        pos=np.arange(1,len(rank_tmp)+1)
        AP.append(np.sum(ap/pos)/n)
        i += 1
    return np.mean(AP)




#VECCHIE FUNZIONI

#Prende in ingresso i ranks e calcola la cmc.
#def calculateCmcFromRanks(ranks,id_probes):
#    cmc=np.zeros((len(ranks)))
#    i=0
#    for p_id in id_probes:
#        # Per tenere conto di un problema multi-shot filtro con >=1
#        cmc += ((ranks[:,i]==p_id).cumsum()) >= 1 
#        i += 1
#    return cmc/len(id_probes)
#     
##Prende in ingresso i ranks e calcola mAP. 
##Il paramentro k indica l'ultima posizione in cui calcolo AP.
#def calculate_mAP(ranks,id_query,k):
#    AP=[]
#    pos=np.arange(1,len(ranks)+1)
#    i=0
#    for q in id_query:
#        rank_tmp=ranks[:,i]
#        #Calcolo il numero di istanze n di q in gallery
#        n=np.sum(rank_tmp==q)
#        ap=np.cumsum(rank_tmp==q)*(rank_tmp==q)        
#        AP.append(np.sum(ap[0:k]/pos[0:k])/n)
#        i += 1
#    return np.mean(AP)



    
def test_camB_vs_camA():
    
    id_probes=Id_B
    probes=[histogram_vector(i) for i in camB]
    
    id_gallery=Id_A 
    gallery=[histogram_vector(i) for i in camA]
    
    cmc_vector,positions=cmc(probes, id_probes, gallery, id_gallery)
    plot_CMC(cmc_vector)  
    print(sorted(positions))
     
        

def test_11B_vs_allA():
    id_probes=[Id_B[i] for i in range(0,110,10)] 
    set_of_probes=[histogram_vector(camB[i]) for i in range(0,110,10)]
    
    id_gallery=Id_A
    gallery=[histogram_vector(i) for i in camA]
        
    cmc_vector,positions= cmc(set_of_probes, id_probes, gallery, id_gallery)
    plot_CMC(cmc_vector)  
    print(sorted(positions))
   
       
if __name__ == '__main__':
    
    r1=np.array([1,0,1,0,0,1,0,0,1,1,0,1,0,1,1])
    
    r=r1.reshape((5,3))
    q=[1,0,1]
    m1=calculate_mAP(r,q,len(r))
    m2=calculate_mAP2(r,q,len(r))
    
    
#    print(calculateCmcFromRanks(r,q))
#    print(calculateCmcFromRanks2(r,q))
    
#    camA,Id_A,camB,Id_B=loadVIPeR(False)
#    start=time.time()

    print("START!")
    
    #test_camB_vs_camA()
#    test_11B_vs_allA()
    
    

    
    
    

    






 
    