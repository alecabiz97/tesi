# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:02:19 2020

@author: AleCabiz
"""

import os
import glob
from histogram import *
import imageio
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
import pickle


def importFiles(Dir,filesExt):
    images,id_images=[],[]
    for el in os.scandir(Dir):
        if el.name.endswith(filesExt) and el.is_file():
            image=imageio.imread(os.path.join(Dir,el.name))
            images.append(image)
            image_id=int(el.name.split('_')[0])
            id_images.append(image_id)          
    return images,id_images


def loadVIPeR():
    DircamA='..\\VIPeR\\cam_a'
    DircamB='..\\VIPeR\\cam_b'
    
    galleryA, idA=importFiles(DircamA,'.bmp')
    galleryB, idB=importFiles(DircamB,'.bmp')
    return galleryA,idA,galleryB,idB

def loadMarket_1501():
    Dir_test='..\\Market-1501\\bounding_box_test'
    Dir_train='..\\Market-1501\\bounding_box_train'
    Dir_query='..\\Market-1501\\query'
    
    test, id_test=t=importFiles(Dir_test,'.jpg')
    train, id_train=t=importFiles(Dir_train,'.jpg')
    query, id_query=importFiles(Dir_query,'.jpg')
    
    return [(test,train,query),(id_test,id_train,id_query)]

def loadDukeMTMC_reID():
    Dir_test='..\\DukeMTMC-reID\\bounding_box_test'
    Dir_train='..\\DukeMTMC-reID\\bounding_box_train'
    Dir_query='..\\DukeMTMC-reID\\query'
    
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

def loadCNN(Dir):
    featureCnn=[]
    for el in os.scandir(Dir):
        if el.name.endswith('.pkl') and el.is_file():
            Dir_file=os.path.join(Dir,el.name)
            file=open(Dir_file,'rb')
            featureCnn.append(pickle.load(file))
    test,query,train=featureCnn[0:4],featureCnn[4:8],featureCnn[8:12]
    return  test,query,train 

def saveFile(filename,X):
    with open(filename, 'wb') as f:
        pickle.dump(X, f)
        
def loadFile(filename):
    with open(filename, 'rb') as f:
        X=pickle.load(f)         
    return X
   
    
if __name__ == '__main__': 

    #Load VIPeR
    CamA,Id_A,CamB,Id_B=loadVIPeR() 

#    
#    
#   #Load Market-1501
#    gallery,ID=loadMarket_1501()
#    
#    #Load DukeMTMC_reID
#    gallery,ID=loadDukeMTMC_reID()
#    
#    test0, train0, query0 = gallery
#    id_test0, id_train0, id_query0 = ID
#    
  
    #
    print('Dataset importato')
    
                    