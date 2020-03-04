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

#KeysA e KeysB conterrano le chievi dei dizionari camA e camB
KeysA,KeysB=[],[]
for k in camA.keys():
    KeysA.append(k) #guardare cosa è k, le chiavi è meglio che siano interi
for k in camB.keys():
    KeysB.append(k)    
    #keys -> id_a
    
V=reshape_image(camA[KeysA[0]])    
histogram_plot(V)