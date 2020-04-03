# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:55:51 2020

@author: AleCabiz
"""
import numpy as np
from histogram import *
import time
from skimage import color
from skimage import feature



#LBP() rieceve in input una matrice e restituisce la matrice LBP(Local Binary Pattern)
def LBP(X):
    riga,colonna,index=0,0,0
    n_pix=(X.shape[0]*X.shape[1])-(2*X.shape[0] + (2*X.shape[1] - 4)) #(128*48)-(2*128+46+46) tutti i pixel tranne il bordo
    matrixLbp=np.zeros((X.shape[0]-2,X.shape[1]-2),int) 
    while riga < (X.shape[0]-2): 
        colonna = 0
        while colonna < (X.shape[1]-2):
            w=np.zeros((3,3))
            w=X[riga:riga+3,colonna:colonna+3].copy()
            lbp_value=window3x3Value(w)
            matrixLbp[riga,colonna]=lbp_value 
            index +=1
            colonna += 1
        riga += 1
    return matrixLbp

def window3x3Value(w):
    riga_tmp,colonna_tmp=0,0
    lbp_binary_value=''
    while riga_tmp < 3 or colonna_tmp < 3:
        colonna_tmp = 0
        while colonna_tmp < 3:
            if riga_tmp == 1 and colonna_tmp == 1:
                colonna_tmp += 1
            if w[riga_tmp,colonna_tmp] >= w[1,1]:
                lbp_binary_value += '1'
            else:
                lbp_binary_value += '0'
            colonna_tmp += 1
        riga_tmp += 1 
    return int(lbp_binary_value,2) #converto da binario a decimale

#LbpRGB riceve in ingresso un immagine RGB e restituisce la matrice LBP, facendo prima una conversione a livelli di grigio.
def LbpRGB(X):
    return LBP(color.rgb2grey(X))


#Lbp3Channel riceve in ingresso un immagine RGB e restituisce 3 istogrammi LBP, 1 per canale
def histogram_Lbp3Channel(X):
    Xr=X[:,:,0]
    Xg=X[:,:,1] 
    Xb=X[:,:,2]     
    lbp_xr=histogram_vector(LBP(Xr))
    lbp_xg=histogram_vector(LBP(Xg))
    lbp_xb=histogram_vector(LBP(Xb))
    return np.concatenate((lbp_xr,lbp_xg,lbp_xb))

if __name__ == '__main__':
    
    #Prova LBP
    #Prendo le prime n immagini di camA e per ciscuna confronto l'istogramma LBP
    #con un numero m di immagini in camB. In result i TRUE indicano che ki>kj(ki->stessa persona,
    #kj->persone diverse) dove k è l'indice di similarità.Confronto il primo canale
    
    
    start=time.time()
    X=img
  #  X=color.rgb2grey(camA[0])
#    h=LBP(X)
#    x=np.arange(0,256,1)
#    
#    pl.plot(x,h)
    X2=LBP(X)
    pl.imshow(X)
    pl.show()
    pl.imshow(X2)
    pl.show()
    
    
    
    end=time.time()
    tempo=end-start
    print(tempo)
    
    
    
              
             
