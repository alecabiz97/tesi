# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:53:42 2020

@author: AleCabiz
"""
import numpy as np
from histogram import *


#HOG() rieceve in input una matrice e restituisce l'istogramma HOG(Histogram of oriented gradients),un vettore colonna di 15x5x36=2700 elementi
#X è 128*48, divido in blocchi da 16. Per ciscun blocco considero 4 blocchi 8*8 e per ciscuno calcolo HOG da 9 bin.
#Quindi per ogni blocco da 16 concateno 4 histogrammi ottenendo un vettore da 36, che verra normalizzato.
#Poi mi sposto di 8 colonne e itero il processo sino alla fine della riga. Incremento la riga fino alla fine dell immagine . 
def HOG(X):
    riga,colonna=0,0  #indici di X
    hog=[]
    while riga <= X.shape[0]-16: #128-16=112
        colonna=0
        while colonna <= (X.shape[1]) -16: #48-16=32
            W_16=np.zeros((16,16))
            W_16=X[riga:riga+16,colonna:colonna+16].copy()
            riga_W16,colonna_W16=0,0   #indici della finestra 16x16
            hog16=[]
            while riga_W16 < 16:
                colonna_W16 = 0
                while colonna_W16 < 16:
                    W_8=np.zeros((8,8))
                    W_8=W_16[riga_W16:riga_W16+8,colonna_W16:colonna_W16+8]
                    riga_tmp,colonna_tmp=0,0  #indici della finestra 8x8
                    vectorHog_tmp=[]  #conterrà magnitude e direction della finestra 3x3
                    while riga_tmp < 6:
                        colonna_tmp=0
                        while colonna_tmp < 6:
                            w=np.zeros((3,3))
                            w=W_8[riga_tmp:riga_tmp+3,colonna_tmp:colonna_tmp+3].copy()
                            x_direction=np.abs(int(w[1,0])-int(w[1,2]))
                            y_direction=np.abs(int(w[0,1])-int(w[2,1]))
                            grad_mag=np.sqrt((x_direction**2)+(y_direction**2)) #calcolo la gradient magnitude (modulo) 
                            if y_direction == 0:        #np.arctan() di infinito da errore
                                grad_direction = 90   #calcolo la gradient direction (fase)
                            else:    
                                grad_direction=np.degrees((np.arctan(x_direction/y_direction))) #np.degrees converte in gradi
                            vectorHog_tmp.append([round(grad_mag),round(grad_direction)])
                            colonna_tmp += 1
                        riga_tmp += 1
                    hog8=histogramHOG(vectorHog_tmp)  
                    hog16.append(hog8)
                    colonna_W16 += 8
                riga_W16 += 8
                
            #Normalizzo un blocco da 16
            hog16=np.array(hog16).reshape(np.size(hog16),1) #Converto hog16 da una lista di array ad un array colonna
            tot=np.sqrt(np.sum(np.power(hog16,2)))
            if tot != 0:
                hog16=hog16/int(tot)
            hog.append(hog16)
            colonna += 8
        riga += 8
    hog=np.array(hog)
    hog=hog.reshape(hog.shape[0]*hog.shape[1]) #hog è un vettore colonna di 15x5x36=2700
    return hog

#Dato v una lista di liste di 2 elementi (magnitude,direction) restituisce un vettore di 9 righe.
#Le righe indicano 0°,20°,40°,60°,80°,100°,120°,140°,160° e i valori le magnitude corrispondenti
#Se la direction non è esattamente uno dei numeri sopracittati la corrispondente magnitude viene divisa
#Es. d=35° m=20 -> 1/4 va in 20* e 3/4 in 40*      
def histogramHOG(v):
    h=np.zeros((9,))
    for val in v:
        m,d=val[0],val[1]
        if d == 0:
            h[0] += m
        elif d == 20:
            h[1] += m
        elif d == 40:
            h[2] += m
        elif d == 60:
            h[3] += m
        elif d == 80:
            h[4] += m
        elif d == 100:
            h[5] += m
        elif d == 120:
            h[6] += m
        elif d == 140:
            h[7] += m
        elif d == 160:
            h[8] += m
        elif 0 < d < 20:
            h[0] += (1-(np.abs(d-0)/20))*m
            h[1] += (1-(np.abs(d-20)/20))*m
        elif 20 < d < 40:
            h[1] += (1-(np.abs(d-20)/20))*m
            h[2] += (1-(np.abs(d-40)/20))*m
        elif 40 < d < 60:
            h[2] += (1-(np.abs(d-40)/20))*m
            h[3] += (1-(np.abs(d-60)/20))*m
        elif 60 < d < 80:
            h[3] += (1-(np.abs(d-60)/20))*m
            h[4] += (1-(np.abs(d-80)/20))*m
        elif 80 < d < 100:
            h[4] += (1-(np.abs(d-80)/20))*m
            h[5] += (1-(np.abs(d-100)/20))*m
        elif 100 < d < 120:
            h[5] += (1-(np.abs(d-100)/20))*m
            h[6] += (1-(np.abs(d-120)/20))*m
        elif 120 < d < 140:
            h[6] += (1-(np.abs(d-120)/20))*m
            h[7] += (1-(np.abs(d-140)/20))*m
        elif 140 < d < 1600:
            h[7] += (1-(np.abs(d-140)/20))*m
            h[8] += (1-(np.abs(d-160)/20))*m
        elif 160 < d < 180:
            h[8] += (1-(np.abs(d-160)/20))*m
            h[0] += (1-(np.abs(d-180)/20))*m
    return h

#Hog3Channel riceve in ingresso un immagine RGB e restituisce 3 istogrammi hog, 1 per canale
def Hog3Channel(X):
    Xr=X[:,:,0]
    Xg=X[:,:,1] 
    Xb=X[:,:,2]     
    hog_xr=HOG(Xr)
    hog_xg=HOG(Xg)
    hog_xb=HOG(Xb)
    #hog_tmp=np.maximum(hog_xr,hog_xg)
    return [hog_xr,hog_xg,hog_xb]



if __name__ == '__main__':
        
    #Prova HOG
    #Prendo le prime n immagini di camA e per ciscuna confronto l'istogramma HOG
    #con un numero m di immagini in camB. In result i TRUE indicano che ki>kj(ki->stessa persona,
    #kj->persone diverse) dove k è l'indice di similarità ottenuto facendo la media tra i 3 k calcolati per i 3 canali.
    n=5
    m=20
    for i in range(n):
        Ai=camA[i]
        Bi=camB[i]
        hog_aR, hog_aG, hog_aB = Hog3Channel(Ai)
        hog_bR, hog_bG, hog_bB = Hog3Channel(Bi)

        
        kR=histogram_intersection(hog_aR,hog_bR)
        kG=histogram_intersection(hog_aG,hog_bG)
        kB=histogram_intersection(hog_aB,hog_bB)
        
        ki=(kR + kG + kB)/3
        p=[]
        
        for j in range(m):
            Bj=camB[j]
            hog_bR_j, hog_bG_j, hog_bB_j = Hog3Channel(Bj)
                
            kR_j=histogram_intersection(hog_aR,hog_bR_j)
            kG_j=histogram_intersection(hog_aG,hog_bG_j)
            kB_j=histogram_intersection(hog_aB,hog_bB_j)
            kj=(kR_j + kG_j + kB_j)/3
            p.append(ki>kj)
    
        print('True:' + str(p.count(True)))
        print('False:' + str(p.count(False)))
        print('#########################')

#%%
        A0=camA[0][:,:,0]
        v=HOG(2*A0)
       # v1,v2,v3=Hog3Channel(2*A)
    
