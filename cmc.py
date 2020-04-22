# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:08:24 2020

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
from Lbp import *
from hog import *
import random
import time
from skimage import feature

#Cmc riceve direttamente i vettori delle feature
def cmc(hist_probes, id_probes, hist_gallery, id_gallery):
    
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
        
  
        val_cmc=np.sum(all_ranks==id_probes,1).cumsum()
    return val_cmc/len(id_probes)

#Plot CMC
def plot_CMC(cmc_vector):
    x=np.arange(len(cmc_vector))+1
    pl.plot(x,cmc_vector)
    pl.title('Cumulative Match Characteristic')
    pl.ylabel('Probability of Identification')
    pl.xlabel('Rank')
    pl.grid(True)
    pl.show()  
    
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

    start=time.time()

    print("START!")
    
    #test_camB_vs_camA()
    test_11B_vs_allA()
    
    end=time.time()
    tempo=end-start
    print('Tempo:' + str(tempo)) 
    
    
    

    






 
    