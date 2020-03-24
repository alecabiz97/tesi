# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:07:42 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl

from PIL import Image
import time


def histFromGrayImage(img):
    nBins=256
    histVal=np.zeros(nBins) 
    for i in img.flatten():
        histVal[i]+=1
    return histVal


def histogram_vector(img, numLayer=3):
    nBins=256
    histValRGB=np.zeros(nBins*numLayer)
    for i in range(numLayer):
        i_start=i*nBins
        i_stop=(i+1)*nBins
        histValRGB[i_start: i_stop ]= histFromGrayImage(img[:,:,i])    
    return histValRGB  





def reshape_image(X):     #restituisco 3 array(in caso RGB)
    #r,g,b = img.split()    
    V=[]
    if len(X.shape) == 2:   #greyscale image  
        x=X.reshape([1,X.shape[0]*X.shape[1]]) #anzi che (1,16) (16,1)
        x=np.transpose(x)
        return x
    if len(X.shape) == 3:  #RGB
        for ch in range(X.shape[2]):  #ch anzi l
            X_ch=X[:,:,ch]
            
            #creo un vettore x1 con tutti i valori della matrice Xl
            x=X_ch.reshape([X_ch.shape[0]*X_ch.shape[1],1])
            V.append(x)
        return V[0],V[1],V[2]
    

#inutile
def histogram_plot(V):
    #Plotting
    if np.shape(V)[0] == 3:   #RGB
        pl.subplot(3,1,1)    
        pl.hist(V[0],bins=256, color='r')
        pl.subplot(3,1,2)    
        pl.hist(V[1],bins=256, color='g')
        pl.subplot(3,1,3)  
        pl.hist(V[2],bins=256, color='b')  
        pl.show()
    elif np.shape(V)[0] == 1:  #greyscale image 
        pl.hist(V[0],bins=256)  
        pl.show()

#Prima bozza        
def histogram_intersection(A,B):    #A e B sono due vettori
    if len(A) == len(B):
        return np.sum(np.minimum(A,B))/np.sum(A) 
    else:
        return None
        
     

if __name__ == '__main__':
    
   A=camA[0]
   ha=histogram_vector(A)
        
        
                      
        

                        