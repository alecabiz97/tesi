# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:45 2020

@author: AleCabiz
"""

import os
import glob
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importBmp import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
import time
import random

fileDir='C:\\Users\\AleCabiz\\Desktop\\Tesi\\Python files'
inDir='C:\\Users\\AleCabiz\\Desktop\\Tesi'
newDir=os.path.join(inDir,'VIPeR')
os.chdir(newDir)
filt=glob.glob('cam*')

d_list=importBmpFiles(newDir,filt)
#os.chdir(baseDir) #torno alla directory corrente
camA=d_list[0]
camB=d_list[1]

#Id_A e Id_B conterrano le chievi dei dizionari camA e camB
Id_A,Id_B=[],[]
for k in camA.keys():
    Id_A.append(k) #guardare cosa è k, le chiavi è meglio che siano interi
for k in camB.keys():
    Id_B.append(k)  
    
    

#Test CMC
#Per 10 volte seleziono in immagine a caso in camA calcolo la CMC_curve e alla fine
#faccio il plot delle 10 curve e di quella media    
start=time.time()

cmc_mean_vector=np.zeros((len(Id_B),))  
for i in range(10): 
    rand_image=random.choice(Id_A)
    
    query={rand_image:camA[rand_image]}  
    gallery={k:v for k,v in camB.items()if k < max(Id_B)+1 } # +1 altrimenti non considero l'ultima immagine, non uso len(Id_B) perchè l'ultima immagine ha id 873 invece di 631
    
    #CMC_curve() restituisce la posizione in cui è stata trovata la query e le probabilità di identificazione
    maching_position,cmc_vector=CMC_curve(query,gallery) 
    #Ricavo cmc_mean_vector, alla fine del ciclo verrà diviso per il numero di query scelto
    cmc_mean_vector += cmc_vector
    
    x=np.arange(len(cmc_vector))+1
    pl.plot(x,cmc_vector,linewidth=0.5)
    pl.xlim(1,len(Id_B))
cmc_mean_vector = cmc_mean_vector/(i+1)
pl.plot(x,cmc_mean_vector,linewidth=2)
pl.title('Cumulative Match Characteristic')
pl.ylabel('Probability of Identification')
pl.xlabel('Rank')
pl.show()


end=time.time()
tempo=end-start
print('Tempo:' + str(tempo))  


end=time.time()
tempo=end-start
print('Tempo:' + str(tempo))    

#%%
#Test color, hog, lbp combinati
#n=5
#m=50
#for i in range(n):
#    Ai=camA[i]
#    Bi=camB[i]
#    #Color
##    hist_aR, hist_aG, hist_aB = histogram_vector(Ai)
##    hist_bR, hist_bG, hist_bB = histogram_vector(Bi)
##    
##    kR=histogram_intersection(hist_aR,hist_bR)
##    kG=histogram_intersection(hist_aG,hist_bG)
##    kB=histogram_intersection(hist_aB,hist_bB)
##    ki_color=(kR + kG + kB)/3 
#
#    
#    #LBP
#    Lbp_aR, Lbp_aG, Lbp_aB = Lbp3Channel(Ai)
#    Lbp_bR, Lbp_bG, Lbp_bB = Lbp3Channel(Bi)
#    
#    kR_LBP=histogram_intersection(Lbp_aR,Lbp_bR)
#    kG_LBP=histogram_intersection(Lbp_aG,Lbp_bG)
#    kB_LBP=histogram_intersection(Lbp_aB,Lbp_bB)
#    ki_LBP=(kR_LBP + kG_LBP + kB_LBP)/3
#    
#    #HOG
#    hog_aR, hog_aG, hog_aB = Hog3Channel(Ai)
#    hog_bR, hog_bG, hog_bB = Hog3Channel(Bi)
#
#    kR_HOG=histogram_intersection(hog_aR,hog_bR)
#    kG_HOG=histogram_intersection(hog_aG,hog_bG)
#    kB_HOG=histogram_intersection(hog_aB,hog_bB)
#    ki_HOG=(kR_HOG + kG_HOG + kB_HOG)/3
#
#
#    ki=ki_LBP + ki_HOG
##    ki=ki_color + ki_HOG
##    ki=ki_color + ki_LBP + ki_HOG
#    
#    p=[]
#    
#    for j in range(m):
#        Bj=camB[j]
#        
#        #Color
##        hist_bR_j, hist_bG_j, hist_bB_j = histogram_vector(Bj)
##        kR_j=histogram_intersection(hist_aR,hist_bR_j)
##        kG_j=histogram_intersection(hist_aG,hist_bG_j)
##        kB_j=histogram_intersection(hist_aB,hist_bB_j)
##        kj_color=(kR_j + kG_j + kB_j)/3
#        
#        #LBP
#        Lbp_bR_j, Lbp_bG_j, Lbp_bB_j = Lbp3Channel(Bj)
#                
#        kR_j_LBP=histogram_intersection(Lbp_aR,Lbp_bR_j)
#        kG_j_LBP=histogram_intersection(Lbp_aG,Lbp_bG_j)
#        kB_j_LBP=histogram_intersection(Lbp_aB,Lbp_bB_j)
#        kj_LBP=(kR_j_LBP + kG_j_LBP + kB_j_LBP)/3
#        
#        #HOG
#        hog_bR_j, hog_bG_j, hog_bB_j = Hog3Channel(Bj)
#                
#        kR_j_HOG=histogram_intersection(hog_aR,hog_bR_j)
#        kG_j_HOG=histogram_intersection(hog_aG,hog_bG_j)
#        kB_j_HOG=histogram_intersection(hog_aB,hog_bB_j)
#        kj_HOG=(kR_j_HOG + kG_j_HOG + kB_j_HOG)/3
#            
#        
#        kj=kj_LBP + kj_HOG
##        kj=kj_color + kj_HOG
##        kj=kj_color + kj_LBP + kj_HOG
#        p.append(ki>kj)
#
#    print('True:' + str(p.count(True)))
#    print('False:' + str(p.count(False)))
#    print('#########################')         
#    
#    

















    




