# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:07:42 2020

@author: AleCabiz
"""


import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl

from PIL import Image



def histogram_RGB(X):
    
    #r,g,b = img.split()
    
    V=[]
    for l in range(X.shape[2]):
        X_l=X[:,:,l]
        
        #creo un vettore x1 con tutti i valori della matrice Xl
        x=X_l.reshape([1,X_l.shape[0]*X_l.shape[1]])
        x=np.transpose(x)
        V.append(x)

    #Plotting
    pl.subplot(3,1,1)    
    pl.hist(V[0],bins=256, color='r')
    pl.subplot(3,1,2)    
    pl.hist(V[1],bins=256, color='g')
    pl.subplot(3,1,3)  
    pl.hist(V[2],bins=256, color='b')  
    pl.show()
    return V

if __name__ == '__main__':
    
    histogram_RGB('000_45.bmp')
    


    
                    