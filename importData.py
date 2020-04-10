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




def importFiles(Dir,filesExt):
    images,id_images=[],[]
    os.chdir(Dir)
    filt_files=glob.glob('*' + filesExt) 
    for k in filt_files:
        image=imageio.imread(k)
        images.append(image)
        image_id=int(k.split('_')[0])
        id_images.append(image_id)      
    return images,id_images


def loadVIPeR():
    DircamA='C:\\Users\\AleCabiz\\Desktop\\Tesi\\VIPeR\\cam_a'
    DircamB='C:\\Users\\AleCabiz\\Desktop\\Tesi\\VIPeR\\cam_b'
    
    galleryA, idA=importFiles(DircamA,'.bmp')
    galleryB, idB=importFiles(DircamB,'.bmp')
    return galleryA,idA,galleryB,idB

def loadMarket_1501():
    Dir_test='C:\\Users\\AleCabiz\\Desktop\\Tesi\\Market-1501\\bounding_box_test'
    Dir_train='C:\\Users\\AleCabiz\\Desktop\\Tesi\\Market-1501\\bounding_box_train'
    Dir_query='C:\\Users\\AleCabiz\\Desktop\\Tesi\\Market-1501\\query'
    
    test, id_test=t=importFiles(Dir_test,'.jpg')
    train, id_train=t=importFiles(Dir_train,'.jpg')
    query, id_query=importFiles(Dir_query,'.jpg')
    
    return [(test,train,query),(id_test,id_train,id_query)]

if __name__ == '__main__':            
    
    
    
    
    inDir='C:\\Users\\AleCabiz\\Desktop\\Tesi\\Market-1501\\'
#    newDir=os.path.join(inDir,'')
#    os.chdir(newDir)
#    filt=glob.glob('cam*')
    
    dir_test=os.path.join(inDir,'bounding_box_test')
    dir_train=os.path.join(inDir,'bounding_box_train')
    dir_query=os.path.join(inDir,'query') 
    fileExt='jpg'
        
    gallery_test, id_test=importFiles(dir_test,fileExt)
    gallery_train, id_train=importFiles(dir_train,fileExt)
    gallery_query, id_query=importFiles(dir_query,fileExt)
       
    
    
    
                    