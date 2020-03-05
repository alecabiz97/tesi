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
#histogram_plot(V)
    
V1=histogram_vector(camA[0]) 

#rank=CMC_curve(camA,camB)


#%%

#rank={}
#K=np.zeros((25,2))
#
#Id_A2=Id_A[0:25]
#Id_B2=Id_B[0:25]
#
#for i in Id_A2:
#    A=camA[i]
#    a_r,a_g,a_b=histogram_vector(A)
#    k_index=0
#    for j in Id_B2:
#        B=camB[j]
#        b_r,b_g,b_b=histogram_vector(B)
#        
#        kr=histogram_intersection(a_r,b_r) 
#        kg=histogram_intersection(a_g,b_g) 
#        kb=histogram_intersection(a_b,b_b) 
#        k=(kr+kg+kb)/3 
#        K[k_index]=(j,k)
#        k_index += 1
#    rank[i]=K
#    print(i) 
#    
##%%
#v=np.zeros(25,1)
#v_index,v_index_tmp=0,0
#
#
#for k in rank.keys():
#    K=rank[k]
#    for i in K[:,0]:
#        R=K[i,:]
#        if R[i,1] > v[v_index]:
#            v[v_index]=R[i,:]
#            
            
        
        

















    




