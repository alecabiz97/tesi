# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:28:01 2020

@author: AleCabiz
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from importBmp import *
from histogram import *
from cmc import *
from Lbp import *
from hog import *
import time
import random

class BayesianModel(object):
    
    def __init__(self,trainingSet,querySet,n_istance):
        self.trainingSet=trainingSet
        self.querySet=querySet
        self.n_istance=n_istance
        
    def train(self):
        P_lqEqualslg=1/len(self.trainingSet)*self.n_istance
        P_lqNotEqualslg= 1-P_lqEqualslg
        
        hist_q=[histogram_vector(q) for q in self.querySet]
        hist_t=[histogram_vector(t) for t in self.trainingSet]

        distance=np.zeros((len(hist_t),len(hist_q)),int)
        column=0
        for q in hist_q:
            d=[]
            for t in hist_t:
                d.append(histogram_distance(q,t))
            distance[:,column]=np.array(d)
            column += 1
         
        M=100
        h,bins=np.histogram(distance,M)
        #distance_range=int(np.max(distance)-np.min(distance))
    
        
        X=np.zeros((np.size(distance),2))
        index=0
        for i in range(len(self.querySet)): 
            for d in distance[:,i].flatten():
                h,bins=np.histogram(distance,M)
                for j in range(len(bins)):
                     if d <= bins[j] or d == np.min(distance):
                         break
                 
                P_d_lqEqualslg=len(hist_t)/h[j-1]
                P_d_lqNotEqualslg= 1 - P_d_lqEqualslg
                
                P_d=(P_d_lqEqualslg*P_lqEqualslg)+(P_d_lqNotEqualslg*P_lqNotEqualslg)
        
                #Teorema di Bayes
                P_lqEqualslg_d=(P_d_lqEqualslg*P_lqEqualslg)/P_d
                X[index,:]=[d,P_lqEqualslg_d]
                index += 1
            self.distanceProbability=X
        
                
if __name__ == '__main__':

                
    trainingSet=camA[0:10]
    querySet=camB[0:10]            
    
    B=BayesianModel(trainingSet,querySet,1)
    B.train()      
          
                
                
                
                
                
                
                
                
                
                
        