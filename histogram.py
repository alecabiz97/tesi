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
        x=X.reshape([X.shape[0]*X.shape[1],1]) #anzi che (1,16) (16,1)
        for i in range(len(x)):
            val=x[i]
            V[val] += 1
        return V
    if len(X.shape) == 3:  #RGB
        V_rgb=[]
        for ch in range(X.shape[2]):  
            X_ch=X[:,:,ch]
            
            #creo un vettore x con tutti i valori della matrice X_ch
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
    
    #Prova histogram_vector
    #Prendo le prime n immagini di camA e per ciscuna confronto l'istogramma RGB
    #con un numero m di immagini in camB. In result i TRUE indicano che ki>kj(ki->stessa persona,
    #kj->persone diverse) dove k è l'indice di similarità.Confronto il primo canale
    n=5
    m=50
    for i in range(n):
        Ai=camA[i]
        Bi=camB[i]
        hist_aR, hist_aG, hist_aB = histogram_vector(Ai)
        hist_bR, hist_bG, hist_bB = histogram_vector(Bi)

        
        kR=histogram_intersection(hist_aR,hist_bR)
        kG=histogram_intersection(hist_aG,hist_bG)
        kB=histogram_intersection(hist_aB,hist_bB)
        
        ki=(kR + kG + kB)/3
        p=[]
        
        for j in range(m):
            Bj=camB[j]
            hist_bR_j, hist_bG_j, hist_bB_j = histogram_vector(Bj)
                
            kR_j=histogram_intersection(hist_aR,hist_bR_j)
            kG_j=histogram_intersection(hist_aG,hist_bG_j)
            kB_j=histogram_intersection(hist_aB,hist_bB_j)
            kj=(kR_j + kG_j + kB_j)/3
            p.append(ki>kj)
    
        print('True:' + str(p.count(True)))
        print('False:' + str(p.count(False)))
        print('#########################')
                      
        

                        