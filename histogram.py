# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:07:42 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl

from PIL import Image



def histogram_vector(X):    
    #r,g,b = img.split()    
    V=[]
    if len(X.shape) == 2:   #greyscale image  
        x=X.reshape([1,X.shape[0]*X.shape[1]])
        x=np.transpose(x)
        V.append(x)
        return V
    if len(X.shape) == 3:  #RGB
        for l in range(X.shape[2]):
            X_l=X[:,:,l]
            
            #creo un vettore x1 con tutti i valori della matrice Xl
            x=X_l.reshape([1,X_l.shape[0]*X_l.shape[1]])
            x=np.transpose(x)
            V.append(x)
        return V
    


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
        

if __name__ == '__main__':
    
    histogram_vector('000_45.bmp')
    


    
                    