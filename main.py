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
from importBmp import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
from BayesianModel import *
import time
import random

#Load VIPeR
#camA,idA,camB,idB=loadVIPeR()

#Load Market-1501
gallery,ID=loadMarket_1501()
test, train, query = gallery
id_test, id_train, id_query = ID

print('Dataset importato')

start=time.time()
print('START')

#TEST
#####################################

test_random=np.random.permutation(id_test)
query_random=np.random.permutation(id_query)

test_random,query_random=test_random[0:100],query_random[0:30]

#Feature vector
gallery_test=[histogram_vector(test[i]) for i in test_random]
query_test=[histogram_vector(query[i]) for i in query_random]
#ID
gallery_id_test=[id_test[i] for i in test_random]
query_id_test=[id_query[i] for i in query_random]

print('Test calcolato')
#########################################


#TRAINING
#####################################

i_random=np.random.permutation(id_train)
query_index,train_index=np.split(i_random,2)
#query_index,train_index=i_random[0:30],i_random[30:80]

#Feature vector
gallery_train=[histogram_vector(train[i]) for i in train_index]
query_train=[histogram_vector(train[i]) for i in query_index]
#ID
gallery_id_train=[id_train[i] for i in train_index]
query_id_train=[id_train[i] for i in query_index]

print('Training calcolato')
#########################################


print('START BQE')

B=BayesianModel()
B.train(gallery_train,gallery_id_train,query_train,query_id_train) 

r=[]
for i in range(2):
    rank=B.test(query_test,gallery_test)
    query_test=queryExpansion(rank,gallery_test,query_test,B,5) 
    r.append(rank[0])
end=time.time()
tempo=end-start
print('Tempo:' + str(tempo)) 

 


   


    




