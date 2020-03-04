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

V=reshape_image(camA[0])    
histogram_plot(V)
    
V1=histogram_vector(camA[0]) 
print('ciao')
rank=CMC_curve(camA,camB)

#K=0
#
#for id in Id_A: 
#    B=camB[id]
#    Vb=histogram_vector(B)
#    kR=histogram_intersection(V1[0],Vb[0]) #Red
#    kG=histogram_intersection(V1[1],Vb[1]) #Green
#    kB=histogram_intersection(V1[2],Vb[2]) #Blue
#    k_media=(kR+kG+kB)/3
#    if k_media > K:
#        K=k_media
#        pers=id

#%%
#print(a)
#A=camA[0]
#B=camA[1]
#Va=histogram_vector(A)
#Vb=histogram_vector(B)
#
#kr=histogram_intersection(Va[0],Vb[0]) #11327
#kg=histogram_intersection(Va[1],Vb[1]) #11327
#kb=histogram_intersection(Va[2],Vb[2]) #11327
#K1=(kr+kg+kr)/3 #11327
#
#kr=histogram_intersection(Va[0],Va[0]) #18432
#kg=histogram_intersection(Va[1],Va[1]) #18432
#kb=histogram_intersection(Va[2],Va[2]) #18432
#K2=(kr+kg+kr)/3 #18432
#
#a=Va[0:10]
#b=Vb[0:10]



