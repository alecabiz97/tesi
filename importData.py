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
import shelve

def importFiles(Dir,filesExt):
    images,id_images=[],[]
    for el in os.scandir(Dir):
        if el.name.endswith(filesExt) and el.is_file():
            image=imageio.imread(os.path.join(Dir,el.name))
            images.append(image)
            image_id=int(el.name.split('_')[0])
            id_images.append(image_id)          
    return images,id_images


def loadVIPeR(feature=True):
    if feature:
        filename='..\\VIPeR_histogramRGB.pkl'
        return loadFile(filename)
    else:
        DircamA='..\\VIPeR\\cam_a'
        DircamB='..\\VIPeR\\cam_b'
        
        galleryA, idA=importFiles(DircamA,'.bmp')
        galleryB, idB=importFiles(DircamB,'.bmp')
        return galleryA,idA,galleryB,idB

def loadMarket_1501(feature=True):
     if feature:
        filename='..\\Market-1501_histogramRGB.pkl'
        return loadFile(filename)
     else:
        Dir_test='..\\Market-1501\\bounding_box_test'
        Dir_train='..\\Market-1501\\bounding_box_train'
        Dir_query='..\\Market-1501\\query'
        
        test, id_test=t=importFiles(Dir_test,'.jpg')
        train, id_train=t=importFiles(Dir_train,'.jpg')
        query, id_query=importFiles(Dir_query,'.jpg')
        
        return [(test,train,query),(id_test,id_train,id_query)]

def loadDukeMTMC_reID(feature=True):
     if feature:
        filename='..\\DukeMTMC-reID_histogramRGB.pkl'
        return loadFile(filename)
     else:
        Dir_test='..\\DukeMTMC-reID\\bounding_box_test'
        Dir_train='..\\DukeMTMC-reID\\bounding_box_train'
        Dir_query='..\\DukeMTMC-reID\\query'
        
        test, id_test=t=importFiles(Dir_test,'.jpg')
        train, id_train=t=importFiles(Dir_train,'.jpg')
        query, id_query=importFiles(Dir_query,'.jpg')

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

#    saveFile('B_Market_trained.pkl',B_Market)
#
#    database = shelve.open('B_Market_trained.db') 
#    B=BayesianModel()
#    database['B_Market_trained'] = B
    
    #Load VIPeR
#    camA,Id_A,camB,Id_B=loadVIPeR(False) 

#    
#    
#   #Load Market-1501
#    gallery,ID=loadMarket_1501(False)
#    
#    #Load DukeMTMC_reID
#    gallery,ID=loadDukeMTMC_reID()
#    
#    test0, train0, query0 = gallery
#    id_test0, id_train0, id_query0 = ID
#    
  
    #
    print('Dataset importato')
    
                    