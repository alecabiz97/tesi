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
print('Dataset importato')

start=time.time()

print('START')
    
#TRAINING

#indeces=np.random.permutation(range(len(train)))
indeces=np.arange(0,len(query),1)
n=2000

#Training histogram 
hist_train=[histogram_vector(query[i]) for i in indeces[0:n]]
id_t=[id_query[i] for i in indeces[0:n]]


print('HISTOGRAM COMPUTED')
print('START TRAINING')

B=BayesianModel()
B.train(hist_train,id_t)
print('TRAINING COMPLETE')




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


 

   


    




