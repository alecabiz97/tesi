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


#Load Market-1501
galleryM,IDM=loadMarket_1501()

#Load DukeMTMC_reID
galleryD,IDD=loadDukeMTMC_reID()

print('Dataset importato')
print('START')

testM, trainM, queryM = galleryM
id_testM, id_trainM, id_queryM = IDM


start=time.time()



end=time.time()
tempo=end-start
print('Tempo:' + str(tempo)) 


 

   


    




