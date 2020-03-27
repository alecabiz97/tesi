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
camA_dict=d_list[0]
camB_dict=d_list[1]
 
#Id
Id_A=[i for i in camA_dict.keys()]
Id_B=[i for i in camB_dict.keys()] 
#Immagini
camA=[p for p in camA_dict.values()]
camB=[p for p in camB_dict.values()]

#Test 11 probe da camB e come gallery tutta camA
start=time.time()

id_probes=[Id_B[i] for i in range(0,110,10)] 
set_of_probes=[camB[i] for i in range(0,110,10)]

id_gallery=Id_A
gallery=camA

print("START!")

cmc_vector,positions= cmc(set_of_probes, id_probes, gallery, id_gallery)
print('done!')
plot_CMC(cmc_vector)

end=time.time()
tempo=end-start
print('Tempo:' + str(tempo)) 

 

   


    




