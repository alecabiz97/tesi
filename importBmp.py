# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:02:19 2020

@author: AleCabiz
"""

import os
import glob
#from PIL import Image
import imageio
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl



def importBmpFiles(Dir,filt):
    d_tmp={}
    d_list=[]
    for cartella in filt:            
        os.chdir(os.path.join(Dir,cartella))
        filt_files=glob.glob('*.bmp') 
        for k in filt_files:
            file=imageio.imread(k)
            file_id=int(k.split('_')[0])
            d_tmp[file_id]=file
        d_list.append(d_tmp)
        d_tmp={}
    return d_list
            

if __name__ == '__main__':            
    inDir='C:\\Users\\AleCabiz\\Desktop\\Tesi'
    newDir=os.path.join(inDir,'VIPeR')
    os.chdir(newDir)
    filt=glob.glob('cam*')
    
    d_list=importBmpFiles(newDir,filt)
    cam1=d_list[0]
    cam2=d_list[1]
    K=[]
    for k in cam1.keys():
        K.append(k)
       
    
                    