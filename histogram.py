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

#Dato una matrice i cui valori sono compresi tra 0 e 255 crea l'istogramma
def histogram_vector(img):
    if len(img.shape) == 2:
        nBins=256
        histVal=np.zeros(nBins) 
        for i in img.flatten():
            histVal[i]+=1
        return histVal
    elif len(img.shape) == 3:
        nBins=256
        histValRGB=np.zeros(nBins*3)
        for i in range(3):
            i_start=i*nBins
            i_stop=(i+1)*nBins
            histVal=np.zeros(nBins) 
            for j in img[:,:,i].flatten():
                histVal[j]+=1
            histValRGB[i_start: i_stop ]=histVal  
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

        
def histogram_intersection(A,B):    #A e B sono due vettori
    if len(A) == len(B):
        return np.sum(np.minimum(A,B))/np.sum(A)
    else:
        return None
        
def histogram_distance(A,B):
    if len(A) == len(B):
        return np.sum(np.abs(A-B))/np.sum(A)
    else:
        return None
     

if __name__ == '__main__':
    
    start=time.time()
    
    
    hA=[histogram_vector(i) for i in camA[0:10]]
    hB=[histogram_vector(i) for i in camB]
    
    a=hA[1]
    d1,d2=[],[]
    for b in hB:
        #k=histogram_intersection(a,b)
        d=histogram_distance(a,b)
        d1.append(d)
        
        
    end=time.time()
    tempo=end-start
    print(tempo)
        
        
                      
        

                        