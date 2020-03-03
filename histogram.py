# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:07:42 2020

@author: AleCabiz
"""


import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl

from PIL import Image
import imageio
#from PIL import ImageTk
#import tkinter as tk
#root = tk.Tk()
#immagine_tk = ImageTk.PhotoImage(Image.open('superMario.jpg'))
#tk.Label(root, image=immagine_tk).pack()
#root.mainloop()

X=imageio.imread('superMario.jpg')
#im=Image.open('superMario.jpg')
#r,g,b = im.split()
#le=len(r.histogram())
#r.histogram()
V=[]

for l in range(X.shape[2]):
    X_l=X[:,:,l]
    #x=np.ones(X_l.shape)
    
    #creo un vettore x1 con tutti i valori della matrice Xl
    x=X_l.reshape([1,X_l.shape[0]*X_l.shape[1]])
    x=np.transpose(x)
    #pl.hist(x)
    #pl.show()            
    V.append(x)


#Plotting
pl.subplot(3,1,1)    
pl.hist(V[0],255, facecolor='r')
pl.subplot(3,1,2)    
pl.hist(V[1], facecolor='g')
pl.subplot(3,1,3)  
pl.hist(V[2], facecolor='b')  
pl.show()




    
                    