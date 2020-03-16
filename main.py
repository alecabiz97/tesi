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
        
        

















    




