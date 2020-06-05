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
    images,id_images,cams_images,descriptor_images=[],[],[],[]
    for el in os.scandir(Dir):
        if el.name.endswith(filesExt) and el.is_file():
            image=imageio.imread(os.path.join(Dir,el.name))
            images.append(image)
            image_id=int(el.name.split('_')[0])
            image_cam=int(el.name.split('_c')[1][0])
            image_descriptor=el.name
            id_images.append(image_id)      
            cams_images.append(image_cam)
            descriptor_images.append(image_descriptor)
    return cams_images,images,id_images,descriptor_images

def importFiles2(Dir,filesExt):
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
        
        galleryA, idA=importFiles2(DircamA,'.bmp')
        galleryB, idB=importFiles2(DircamB,'.bmp')
        return galleryA,idA,galleryB,idB

def loadMarket_1501(feature=True):
     if feature:
        filename='..\\Market-1501_histogramRGB.pkl'
        return loadFile(filename)
     else:
        Dir_test='..\\Market-1501\\bounding_box_test'
        Dir_query='..\\Market-1501\\query'
        Dir_train='..\\Market-1501\\bounding_box_train'
        
        cams_test, images_test, id_test, descriptor_test = importFiles(Dir_test,'.jpg')
        cams_query, images_query, id_query, descriptor_query = importFiles(Dir_query,'.jpg')
        cams_train, images_train, id_train, descriptor_train = importFiles(Dir_train,'.jpg')
        
        test=cams_test, images_test, id_test, descriptor_test
        query=cams_query, images_query, id_query, descriptor_query
        train=cams_train, images_train, id_train, descriptor_train
        
        return test,query,train

def loadDukeMTMC_reID(feature=True):
     if feature:
        filename='..\\DukeMTMC-reID_histogramRGB.pkl'
        return loadFile(filename)
     else:
        Dir_test='..\\DukeMTMC-reID\\bounding_box_test'
        Dir_query='..\\DukeMTMC-reID\\query'
        Dir_train='..\\DukeMTMC-reID\\bounding_box_train'
        
        cams_test, images_test, id_test, descriptor_test = importFiles(Dir_test,'.jpg')
        cams_query, images_query, id_query, descriptor_query = importFiles(Dir_query,'.jpg')
        cams_train, images_train, id_train, descriptor_train = importFiles(Dir_train,'.jpg')
        
        test=cams_test, images_test, id_test, descriptor_test
        query=cams_query, images_query, id_query, descriptor_query
        train=cams_train, images_train, id_train, descriptor_train
        
        return test,query,train

def loadCNN(Dir):
    featureCnn=[]
    for el in os.scandir(Dir):
        if el.name.endswith('.pkl') and el.is_file():
            Dir_file=os.path.join(Dir,el.name)
            file=open(Dir_file,'rb')
            featureCnn.append(pickle.load(file))
    test,query,train=featureCnn[0:4],featureCnn[4:8],featureCnn[8:12]
    return  test,query,train 

def loadCNN_CrossDataset(Dir):
    featureCnn=[]
    for el in os.scandir(Dir):
        if el.name.endswith('.pkl') and el.is_file():
            Dir_file=os.path.join(Dir,el.name)
            file=open(Dir_file,'rb')
            featureCnn.append(pickle.load(file))
    return  featureCnn 

def saveFile(filename,X):
    with open(filename, 'wb') as f:
        pickle.dump(X, f)
        
def loadFile(filename):
    with open(filename, 'rb') as f:
        X=pickle.load(f)     
    return X
   
    
if __name__ == '__main__':
    
    
    DirMarket = '..\\FeatureCNN\\Market-1501'
    DirDuke = '..\\FeatureCNN\\DukeMTMC'
    
#    X=loadCNN_CrossDataset('..//FeatureCrossDataset')
#    XDuke,XMarket=X[0:2],X[2::]
#    
#    f=open('DukeFromMarket_feature.pkl','wb')
#    pickle.dump(XDuke,f)
#    f.close()
#    
#    f=open('MarketFromDuke_feature.pkl','wb')
#    pickle.dump(XMarket,f)
#    f.close()
#    
#    f=open('DukeFromMarket_feature.pkl','rb')
#    d=pickle.load(f)
#    f.close()
# 
#    f=open('MarketFromDuke_feature.pkl','rb')
#    m=pickle.load(f)
#    f.close()    
#    
#    print((len(d[0]),len(d[1])))
#    print((len(m[0]),len(m[1])))
    
        
    Dir9='..//FeatureCrossDataset//dukeMTMCfrommarket1501_gallery_descriptors.pkl'
    Dir10='..//FeatureCrossDataset//dukeMTMCfrommarket1501_query_descriptors.pkl'
    Dir11='..//FeatureCrossDataset//market1501fromdukeMTMC_gallery_descriptors.pkl'
    Dir12='..//FeatureCrossDataset//market1501fromdukeMTMC_query_descriptors.pkl'

	#Feature CNN
#	testData,queryData,trainingData=loadCNN(DirDuke)
	
#	test_cams, test_feature, test_id, test_desc = testData
#	query_cams, query_feature, query_id, query_desc = queryData
#	train_cams, train_feature, train_id, train_desc = trainingData
#    a=2
#    X=loadCNN(DirMarket)
#
#    
#    f=open('Market_feature.pkl','wb')
#    pickle.dump(X,f)
#    f.close()


    
                    