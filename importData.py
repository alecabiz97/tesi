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

def loadDukeMTMC_reID():
    Dir_test='C:\\Users\\AleCabiz\\Desktop\\Tesi\\DukeMTMC-reID\\bounding_box_test'
    Dir_train='C:\\Users\\AleCabiz\\Desktop\\Tesi\\DukeMTMC-reID\\bounding_box_train'
    Dir_query='C:\\Users\\AleCabiz\\Desktop\\Tesi\\DukeMTMC-reID\\query'
    
    test, id_test=t=importFiles(Dir_test,'.jpg')
    train, id_train=t=importFiles(Dir_train,'.jpg')
    query, id_query=importFiles(Dir_query,'.jpg')
    
    query1=[[] for i in range(max(id_query)+1)]
    for i in range(len(id_query)):
        query1[id_query[i]].append(query[i])

    test1=[[] for i in range(max(id_test)+1)]
    for i in range(len(id_test)):
        test1[id_test[i]].append(test[i])
        
    train1=[[] for i in range(max(id_train)+1)]
    for i in range(len(id_train)):
        train1[id_train[i]].append(train[i])
        
    #id_test=sorted(set(id_test))
    #id_train=sorted(set(id_train))
    #id_query=sorted(set(id_query))
    
    return [(test,train,query),(id_test,id_train,id_query)]
if __name__ == '__main__':            
    
    
    
    
    #Load VIPeR
#   camA,Id_A,camB,Id_B=loadVIPeR()
    
   #Load Market-1501
    gallery,ID=loadMarket_1501()
    
    #Load DukeMTMC_reID
#    gallery,ID=loadDukeMTMC_reID()
    
    test0, train0, query0 = gallery
    id_test0, id_train0, id_query0 = ID
    
    test, train, query = test0.copy(), train0.copy(), query0.copy()
    id_test, id_train, id_query = id_test0.copy(), id_train0.copy(), id_query0.copy()

    
    #
    print('Dataset importato')
    
                    