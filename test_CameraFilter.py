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
from utils.Lbp import *
from utils.hog import *
from utils.BayesianModel import *
from utils.queryExpansion import *
import pickle
from test_vari.test_FeedbackPesato_Similarity_CameraFilter import *
from test_vari.test_BQE_Similarity_CameraFilter import *
from test_vari.test_FeedbackNonPesato_Similarity_CameraFilter import *
from test_vari.test_AQE_Similarity_CameraFilter import *
from test_vari.test_BQE_CrossDataset_Similarity_CameraFilter import *
from test_vari.test_AQE_CrossDataset_Similarity_CameraFilter import *
from test_vari.test_FeedbackNonPesato_CrossDataset_Similarity_CameraFilter import *
from test_vari.test_FeedbackPesato_CrossDataset_Similarity_CameraFilter import *

#Rocchio
from test_vari.test300_Rocchio import *

#Dataset='Duke'
#Dataset='Market'

D=['Market','Duke']
for Dataset in D:

    ##Similarità
    test_AQE_Similarity(Dataset)
    test_BQE_Similarity(Dataset)
    test_FeedbackPesato_Similarity(Dataset)
    test_FeedbackNonPesato_Similarity(Dataset)

##Scelta dataset
CrossDataset='DukeFromMarket'
CrossDataset='MarketFromDuke'
d=['DukeFromMarket','MarketFromDuke']
for CrossDataset in d:
##Similarità
    test_AQE_CrossDataset_Similarity(CrossDataset)
    test_BQE_CrossDataset_Similarity(CrossDataset)
    test_FeedbackPesato_CrossDataset_Similarity(CrossDataset)
    test_FeedbackNonPesato_CrossDataset_Similarity(CrossDataset)












