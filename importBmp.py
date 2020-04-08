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
            image=imageio.imread(k)
            image_id=int(k.split('_')[0])
            d_tmp[image_id]=image      #la chiave è l'id il valore è l'immagine
        d_list.append(d_tmp)
        d_tmp={}
    return d_list


def load_images(dir_img,fileExt):
    images=[]
    i=0
    for el in os.scandir(dir_img):
        if el.name.endswith(fileExt) and el.is_file():
            i=i+1
        
            img=imageio.imread(os.path.join(dir_img,el.name))
            images.append(img)
    
    id_images=np.arange(1,i+1)
    return images, id_images            

if __name__ == '__main__':            
    
    
    
    
    inDir='C:\\Users\\AleCabiz\\Desktop\\Tesi\\Market-1501\\'
#    newDir=os.path.join(inDir,'')
#    os.chdir(newDir)
#    filt=glob.glob('cam*')
    
    dir_test=os.path.join(inDir,'bounding_box_test')
    dir_train=os.path.join(inDir,'bounding_box_train')
    dir_query=os.path.join(inDir,'query') 
    fileExt='jpg'
        
    gallery_test, id_test=load_images(dir_test,fileExt)
    gallery_train, id_train=load_images(dir_train,fileExt)
    gallery_query, id_query=load_images(dir_query,fileExt)
       
    
    
    
                    