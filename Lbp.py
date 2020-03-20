# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:55:51 2020

@author: AleCabiz
"""
import numpy as np
from histogram import *

#LBP() rieceve in input una matrice e restituisce l'istogramma LBP(Local Binary Pattern)
def LBP(X):
    riga,colonna=0,0
    vectorLbp=[]
    while riga < (X.shape[0]-2): 
        colonna = 0
        while colonna < (X.shape[1]-2):
            w=np.zeros((3,3))
            w=X[riga:riga+3,colonna:colonna+3].copy()
            riga_tmp,colonna_tmp=0,0
            vectorLbp_tmp=[]
            while riga_tmp <=2 or colonna_tmp <= 2:
                colonna_tmp = 0
                while colonna_tmp <= 2:
                    if riga_tmp == 1 and colonna_tmp == 1:
                        colonna_tmp += 1
                    if w[1,1] <= w[riga_tmp,colonna_tmp]:
                        vectorLbp_tmp.append(1)
                    else:
                        vectorLbp_tmp.append(0)
                    colonna_tmp += 1
                riga_tmp += 1
            vectorLbp.append(fromBinToDec(vectorLbp_tmp)) #converto da binario a decimale
            colonna += 1
        riga += 1
    hist_Lbp=histogram_LBP(vectorLbp)
    return hist_Lbp

#Data una lista di boolean restituisce il valore in decimale
def fromBinToDec(lista):
    val=''
    for i in lista:
        val += str(i)
    return int(val,base=2)

#Dato un vettore i cui valori sono compresi tra 0 e 255 crea l'istogramma
def histogram_LBP(v):
    hist_Lbp=np.zeros((256,1))
    for val in v:
        hist_Lbp[val] += 1
    return hist_Lbp


#Lbp3Channel riceve in ingresso un immagine RGB e restituisce 3 istogrammi LBP, 1 per canale
def Lbp3Channel(X):
    Xr=X[:,:,0]
    Xg=X[:,:,1] 
    Xb=X[:,:,2]     
    lbp_xr=LBP(Xr)
    lbp_xg=LBP(Xg)
    lbp_xb=LBP(Xb)
    return lbp_xr,lbp_xg,lbp_xb

if __name__ == '__main__':
    
    #Prova LBP
    #Prendo le prime n immagini di camA e per ciscuna confronto l'istogramma LBP
    #con un numero m di immagini in camB. In result i TRUE indicano che ki>kj(ki->stessa persona,
    #kj->persone diverse) dove k è l'indice di similarità.Confronto il primo canale
    n=5
    m=50
    for i in range(n):
        Ai=camA[i]
        Bi=camB[i]
        Lbp_aR, Lbp_aG, Lbp_aB = Lbp3Channel(Ai)
        Lbp_bR, Lbp_bG, Lbp_bB = Lbp3Channel(Bi)

        
        kR=histogram_intersection(Lbp_aR,Lbp_bR)
        kG=histogram_intersection(Lbp_aG,Lbp_bG)
        kB=histogram_intersection(Lbp_aB,Lbp_bB)
        
        ki=(kR + kG + kB)/3
        p=[]
        
        for j in range(m):
            Bj=camB[j]
            Lbp_bR_j, Lbp_bG_j, Lbp_bB_j = Lbp3Channel(Bj)
                
            kR_j=histogram_intersection(Lbp_aR,Lbp_bR_j)
            kG_j=histogram_intersection(Lbp_aG,Lbp_bG_j)
            kB_j=histogram_intersection(Lbp_aB,Lbp_bB_j)
            kj=(kR_j + kG_j + kB_j)/3
            p.append(ki>kj)
    
        print('True:' + str(p.count(True)))
        print('False:' + str(p.count(False)))
        print('#########################')
              
             
