# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:53:42 2020

@author: AleCabiz
"""
import numpy as np
from histogram import *

def HOG(M,D):
    riga, colonna = 0,0
    lenght_hog=3*9 #Immagine divisa in 3 striscie
    hog=np.zeros(lenght_hog)
    i=0
    while riga < M.shape[0]:
        m,d=np.zeros((42,46)),np.zeros((42,46))
        m=M[riga:riga +42,:].copy()
        d=D[riga:riga +42,:].copy()
        i_start=i*9
        i_end=(i+1)*9
        hog[i_start:i_end]=histogramHOG(m.flatten(),d.flatten())
        i += 1
        riga += 42
#    while riga < M.shape[0]-7:
#        colonna=0
#        while colonna < M.shape[1]-7:
#            
#            m,d=np.zeros((8,8)),np.zeros((8,8))
#            m=M[riga:riga +8,colonna:colonna+8].copy()
#            d=D[riga:riga +8,colonna:colonna+8].copy()
#            i_start=i*9
#            i_end=(i+1)*9
#            hog[i_start:i_end]=histogramHOG(m.flatten(),d.flatten())
#            i += 1
#            colonna += 1
#        riga += 1
    return hog

def calculateMagntiudeDirections(X):
    riga,colonna=0,0   
    new_shape=(X.shape[0]-2,X.shape[1]-2)
    magnitude=np.zeros(new_shape,int)
    directions=np.zeros(new_shape,int)
    while riga < (X.shape[0]-2): 
       colonna = 0
       while colonna < (X.shape[1]-2):
           w=np.zeros((3,3))
           w=X[riga:riga+3,colonna:colonna+3].copy()
           magn,direct=hog3x3window(w)
           magnitude[riga,colonna]=magn
           directions[riga,colonna]=direct
           colonna += 1
       riga += 1
    return magnitude,directions


#Hog3Channel riceve in ingresso un immagine RGB e restituisce 3 istogrammi hog, 1 per canale
def Hog3Channel(X):
#    m,d=[],[]
#    for ch in range(3):
#        mi,di=calculateMagntiudeDirections(X[:,:,ch])
#        m.append(mi)
#        d.append(di)
#    
#    m_max=m[0]
#    direct=d[0]
#    for i in [1,2]:
#        for j in range(m_max.shape[0]):
#            for k in range(m_max.shape[1]):
#                if m[i][j,k] > m_max[j,k]:
#                    m_max[j,k] = m[i][j,k]
#                    direct[j,k] = d[i][j,k]
#                
#    return HOG(m_max, direct)
#    return ft.hog(X,feature_vector=True,multichannel=True)
    
    h=np.zeros(3*27)    
    i=0 
    i_start=i*27
    i_end=(i+1)*27
    for ch in range(3):
        mi,di=calculateMagntiudeDirections(X[:,:,ch])
        h[i_start:i_end]=HOG(mi,di)
        i += 1
    return h
        
        

def hog3x3window(w):
    x_direction=np.abs(int(w[1,0])-int(w[1,2]))
    y_direction=np.abs(int(w[0,1])-int(w[2,1]))
    grad_mag=np.sqrt((x_direction**2)+(y_direction**2)) #calcolo la gradient magnitude (modulo) 
    if y_direction == 0:        #np.arctan() di infinito da errore
        grad_direction = 90   #calcolo la gradient direction (fase)
    else:    
        grad_direction=np.degrees((np.arctan(x_direction/y_direction))) #np.degrees converte in gradi
    return round(grad_mag),round(grad_direction)
    


#Dato v una lista di liste di 2 elementi (magnitude,direction) restituisce un vettore di 9 righe.
#Le righe indicano 0°,20°,40°,60°,80°,100°,120°,140°,160° e i valori le magnitude corrispondenti
#Se la direction non è esattamente uno dei numeri sopracittati la corrispondente magnitude viene divisa
#Es. d=35° m=20 -> 1/4 va in 20* e 3/4 in 40*      
def histogramHOG(magn,direct):
    h=np.zeros(9)
    for m,d in zip(magn,direct):
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




if __name__ == '__main__':
   

    
    A=camA[60]
    h=Hog3Channel(A)
    m,d=calculateMagntiudeDirections(A[:,:,0])
#    h=HOG(m,d)
#    print(len(h))
#    
#    h1=feature.hog(A[:,:,0])
    
    
    
#    ho0,d0 = calculateMagntiudeDirections(A[:,:,0])
#    ho1,d1 = calculateMagntiudeDirections(A[:,:,1])
#    ho2,d2 = calculateMagntiudeDirections(A[:,:,2])
#    
#    h_max=np.maximum(np.maximum(ho0,ho1),ho2)
#    
#    lbp=LbpRGB(A)
#    
#    pl.subplot(1,3,1)
#    pl.imshow(A)
#    pl.subplot(1,3,2)
#    pl.imshow(lbp)
#    pl.subplot(1,3,3)
#    pl.imshow(m)
#    pl.show()
   
   
   
   
   
   
   
   
   
   
   
   
