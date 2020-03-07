# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:07:42 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl

from PIL import Image


def histogram_vector(X):      #restituisce 3 array r,g,b, 1 se levelgrey  
    V=np.zeros([256,1])
    if len(X.shape) == 2:   #greyscale image        
        x=X.reshape([1,X.shape[0]*X.shape[1]]) #anzi che (1,16) (16,1)
        for i in range(len(x)):
            val=x[i]
            V[val] += 1
        return V
    if len(X.shape) == 3:  #RGB
        V_rgb=[]
        for ch in range(X.shape[2]):  #ch anzi l
            X_ch=X[:,:,ch]
            
            #creo un vettore x1 con tutti i valori della matrice Xl
            x=X_ch.reshape(X_ch.shape[0]*X_ch.shape[1],1)
            for i in range(len(x)):
                val=x[i]
                V[val] += 1
            V_rgb.append(V)
        return V_rgb[0],V_rgb[1],V_rgb[2]

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
            x=X_ch.reshape([1,X_ch.shape[0]*X_ch.shape[1]])
            x=np.transpose(x)
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
        return np.sum(np.minimum(A,B))
    else:
        return None
        
     

if __name__ == '__main__':
    
   a=np.array([1,4,3])
   b=np.array([10,2,34])
   print(histogram_intersection(a,b))
   
   a_r,a_g,a_b=histogram_vector(A)
   
    # rank=CMC_curve(camA,camB)
#    A=camA[KeysA[0]]
#    V1=histogram_vector(camA[0])
#    K=0
#    
#    for key in KeysA: 
#        if key != KeysA[0]:
#            B=camA[key]
#            Va=histogram_vector(A)
#            Vb=histogram_vector(B)
#            kR=histogram_intersection(Va[0],Vb[0]) #Red
#            kG=histogram_intersection(Va[1],Vb[1]) #Green
#            kB=histogram_intersection(Va[2],Vb[2]) #Blue
#            k_media=(kR+kG+kB)/3
#            if k_media > K:
#                K=k_media
#                pers=key
        
        

                        