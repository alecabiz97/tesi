# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:45 2020

@author: AleCabiz
"""

import os
import glob
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importData import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
from BayesianModel import *
import time
import random



#Load VIPeR
#camA,Id_A,camB,Id_B=loadVIPeR()

#Load Market-1501
gallery,ID=loadMarket_1501()

#Load DukeMTMC_reID
#gallery,ID=loadDukeMTMC_reID()
#
test, train, query = gallery
id_test, id_train, id_query = ID
##
#print('Dataset importato')

start=time.time()


print('START')

    
#TRAINING

irand=np.random.permutation(range(len(train)))

#n=500

#Training histogram color
hist_train=[histogram_vector(train[i]) for i in irand[0::]]
id_t=[id_train[i] for i in irand[0::]]



#TEST
#m=50
#i_test=np.random.permutation(range(len(test)))
#gallery_test=[histogram_vector(test[i]) for i in i_test[0:m]]
#id_te=[id_test[i] for i in i_test[0:m]]
#
#n_query=3
#i_query=np.random.permutation(range(len(query)))
#query_test=[histogram_vector(query[i]) for i in i_query[0:n_query]]
#id_q=[id_query[i] for i in i_query[0:n_query]]



print('HISTOGRAM COMPUTED')
print('START TRAINING')

B=BayesianModel()
B.train(hist_train,id_t)
print('TRAINING COMPLETE')
print((B.P_ltiEqualsltj,B.P_ltiNotEqualsltj)) 

h1,b1=B.hist_d_sameId
h2,b2=B.hist_d_differentId 

x1=np.zeros_like(h1)
x2=np.zeros_like(h2)
for i in range(len(b1)-1):
    x1[i]=((b1[i] + b1[i+1])/2)
    x2[i]=((b2[i] + b2[i+1])/2)



pl.plot(x1,h1,label='same',color='b')
pl.plot(x2,h2,label='diff',color='r')
 
pl.legend()
pl.show()


#print('START TEST')
#
#r=[]
#for i in range(2):
#    ranks=calculateRanks(query_test,gallery_test,B)
#    print('Ranks calcolato')
#    query_test=queryExpansion(ranks,gallery_test,query_test,3) 
#    print('Nuova query calcolata')
#
#    r.append(ranks[0])


end=time.time()
tempo=end-start
print('Tempo:' + str(tempo)) 


 

   


    




