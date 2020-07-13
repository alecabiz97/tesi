# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:42:39 2020

@author: aleca
"""


import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from BayesianModel import *
from utils.queryExpansion import *
import pickle
from test_vari.test_FeedbackPesato_Similarity import *
from test_vari.test_BQE_Similarity import *
from test_vari.test_FeedbackNonPesato_Similarity import *
from test_vari.test_FeedbackNonPesato import *
from test_vari.test300_FeedbackNonPesato import *
from test_vari.test300_FeedbackPesato import *
from test_vari.test_FeedbackPesato import *
from test_vari.test_AQE import *
from test_vari.test_AQE_Similarity import *
from test_vari.test300_BQE import *
from test_vari.test300_AQE import *
from test_vari.test_BQE import *
from test_vari.test_BQE_CrossDataset_Similarity import *
from test_vari.test_AQE_CrossDataset_Similarity import *
from test_vari.test_FeedbackNonPesato_CrossDataset_Similarity import *
from test_vari.test_FeedbackPesato_CrossDataset_Similarity import *
from test_vari.test_AQE_CrossDataset import *
from test_vari.test_FeedbackNonPesato_CrossDataset import *
from test_vari.test_BQE_CrossDataset import *
from test_vari.test_FeedbackPesato_CrossDataset import *

#Rocchio
from test_vari.test300_Rocchio import *

#Dataset='Duke'
#Dataset='Market'

D=['Market','Duke']
for Dataset in D:

#print('START ALL TEST')
#test300_AQE(Dataset)
#print('TEST ESEGUITO')
#test300_BQE(Dataset)
#print('TEST ESEGUITO')
#test300_FeedbackPesato(Dataset)
#print('TEST ESEGUITO')
#test300_FeedbackNonPesato(Dataset)
#print('TEST ESEGUITO')

#Probabilità
#    test_AQE(Dataset)
#test_BQE(Dataset)
#test_FeedbackPesato(Dataset)
#test_FeedbackNonPesato(Dataset)
#
##Similarità
    test_AQE_Similarity(Dataset)
#test_BQE_Similarity(Dataset)
#test_FeedbackPesato_Similarity(Dataset)
#test_FeedbackNonPesato_Similarity(Dataset)
#
#
##Scelta dataset
#CrossDataset='DukeFromMarket'
#CrossDataset='MarketFromDuke'
#
##Cross Dataset
##Probabilità
#test_AQE_CrossDataset(CrossDataset)
#test_BQE_CrossDataset(CrossDataset)
#test_FeedbackPesato_CrossDataset(CrossDataset)
#test_FeedbackNonPesato_CrossDataset(CrossDataset)
#
##Similarità
#test_AQE_CrossDataset_Similarity(CrossDataset)
#test_BQE_CrossDataset_Similarity(CrossDataset)
#test_FeedbackPesato_CrossDataset_Similarity(CrossDataset)
#test_FeedbackNonPesato_CrossDataset_Similarity(CrossDataset)












