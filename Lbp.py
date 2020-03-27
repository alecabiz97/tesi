# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:55:51 2020

@author: AleCabiz
"""
import numpy as np
from histogram import *
import time



#LBP() rieceve in input una matrice e restituisce l'istogramma LBP(Local Binary Pattern)
def LBP(X):
    riga,colonna,index=0,0,0
    vectorLbp=np.zeros(5796) #(128*48)-(2*128+45+45) tutti i pixel tranne il bordo
    while riga < (X.shape[0]-2): 
        colonna = 0
        while colonna < (X.shape[1]-2):
            w=np.zeros((3,3))
            w=X[riga:riga+3,colonna:colonna+3].copy()
            riga_tmp,colonna_tmp=0,0
            lbp_binary_value=''
            while riga_tmp < 3 or colonna_tmp < 3:
                colonna_tmp = 0
                while colonna_tmp < 3:
                    if riga_tmp == 1 and colonna_tmp == 1:
                        colonna_tmp += 1
                    if w[1,1] <= w[riga_tmp,colonna_tmp]:
                        lbp_binary_value += '1'
                    else:
                        lbp_binary_value += '0'
                    colonna_tmp += 1
                riga_tmp += 1 
            vectorLbp[index]=int(lbp_binary_value,2) #converto da binario a decimale
            index +=1
            colonna += 1
        riga += 1
    hist_Lbp=histogram_LBP(vectorLbp)
    return hist_Lbp



#Dato un vettore i cui valori sono compresi tra 0 e 255 crea l'istogramma
def histogram_LBP(v):
    hist_Lbp=np.zeros(256)
    for val in v:
        hist_Lbp[int(val)] += 1
    return hist_Lbp


#Lbp3Channel riceve in ingresso un immagine RGB e restituisce 3 istogrammi LBP, 1 per canale
def Lbp3Channel(X):
    Xr=X[:,:,0]
    Xg=X[:,:,1] 
    Xb=X[:,:,2]     
    lbp_xr=LBP(Xr)
    lbp_xg=LBP(Xg)
    lbp_xb=LBP(Xb)
    return np.concatenate((lbp_xr,lbp_xg,lbp_xb))

if __name__ == '__main__':
    
    #Prova LBP
    #Prendo le prime n immagini di camA e per ciscuna confronto l'istogramma LBP
    #con un numero m di immagini in camB. In result i TRUE indicano che ki>kj(ki->stessa persona,
    #kj->persone diverse) dove k è l'indice di similarità.Confronto il primo canale
    
    
    start=time.time()
    
    for A in camA[0:50]:
        v=LBP(A[:,:,0])

        
        
    end=time.time()
    tempo=end-start
    print(tempo)
    
    
    
              
             
