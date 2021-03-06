# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:50:37 2020

@author: aleca
"""


import numpy as np
from scipy import linalg
import matplotlib.pyplot as pl
from utils.importData import *
from utils.histogram import *
from utils.evaluation import *
from utils.Lbp import *
from utils.hog import *
import time
import random


def queryExpansion(ranks_index,ranks_probability,gallery,query,K,AQE=False,soglia=0):
    q_expansion=np.zeros_like(query)
    for i in range(len(query)):
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        q_exp=0 
        probability_sum=0
        for j in range(K):
            if AQE:
                probability=1
            else:
                probability=candidates_probability[j]
            if probability > soglia:
                x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
                q_exp += probability*x
                probability_sum += probability 
        q_exp=(q_exp +query[i])/(probability_sum +1)
        q_expansion[i]=q_exp
    return q_expansion


def queryExpansion_withFeedback(ranks_index,ranks_probability,ranks_labels,gallery,query,query_id,K,probEquals1=False,wrongFeed=False):
    q_expansion=np.zeros_like(query)
    for i in range(len(query)):
        q_id=query_id[i]
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        candidates_labels=ranks_labels[:,i][0:K] #Prendo le etichette delle prime K
        q_exp=0 
        probability_sum=0
        if wrongFeed:
            n_distractors=int(K/10) #Ipotizzo di sbagliarne 1 su 10
            cnt_distractor=0
        for j in range(K):
            if q_id == candidates_labels[j]:
                #probEquals1 stabilisce il modo in cui vengono pesate le immagini nella query expansion.
                if not probEquals1:
                    probability=candidates_probability[j]
                else:
                    probability=1
                x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
                q_exp += probability*x
                probability_sum += probability
            elif q_id != candidates_labels[j] and wrongFeed and cnt_distractor<n_distractors:
                if not probEquals1:
                    probability=candidates_probability[j]
                else:
                    probability=1
                x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
                q_exp += probability*x
                probability_sum += probability
                cnt_distractor += 1
                
        q_exp=(q_exp + query[i])/(probability_sum +1)
        q_expansion[i]=q_exp
    return q_expansion

def queryExpansion_withRandomK(ranks_index,ranks_probability,gallery,query,K):
    q_expansion=np.zeros_like(query)
    for i in range(len(query)):
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        q_exp=0 
        probability_sum=0
        k1=np.random.choice(np.arange(1,K)) #Prendo un numero k' di immagini invece di K, k' non può essere uguale a 0
        random_j=np.random.permutation(K)[0:k1]
        for j in random_j: 
            probability=candidates_probability[j]
            x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
            q_exp += probability*x
            probability_sum += probability 
        q_exp=(q_exp +query[i])/(probability_sum +1)
        q_expansion[i]=q_exp    
    return q_expansion




def queryExpansion_withFeedback_CameraFilter(ranks_index,ranks_probability,ranks_labels,gallery,gallery_cams,query,query_id,query_cams,K,probEquals1=False,wrongFeed=False):
    q_expansion=np.zeros_like(query)
    for i in range(len(query)):
        q_id=query_id[i]
        q_cam=query_cams[i]
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        candidates_labels=ranks_labels[:,i][0:K] #Prendo le etichette delle prime K
        candidates_cams=np.array(gallery_cams)[ranks_index[:,i][0:K]]
        q_exp=0 
        probability_sum=0
        if wrongFeed:
            n_distractors=int(K/10) #Ipotizzo di sbagliarne 1 su 10
            cnt_distractor=0
        for j in range(K):
            if q_id == candidates_labels[j] and q_cam != candidates_cams[j]:
                #probEquals1 stabilisce il modo in cui vengono pesate le immagini nella query expansion.
                if not probEquals1:
                    probability=candidates_probability[j]
                else:
                    probability=1
                x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
                q_exp += probability*x
                probability_sum += probability
            elif q_id != candidates_labels[j] and wrongFeed and cnt_distractor<n_distractors:
                if not probEquals1:
                    probability=candidates_probability[j]
                else:
                    probability=1
                x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
                q_exp += probability*x
                probability_sum += probability
                cnt_distractor += 1
                
        q_exp=(q_exp + query[i])/(probability_sum +1)
        q_expansion[i]=q_exp
    return q_expansion


def queryExpansion_CameraFilter(ranks_index,ranks_probability,gallery,gallery_cams,query,query_cams,K,AQE=False,soglia=0):
    q_expansion=np.zeros_like(query)
    for i in range(len(query)):
        candidates_index=ranks_index[:,i][0:K]  #Prendo gli indici dei primi K
        candidates_probability=ranks_probability[:,i][0:K] #Prendo il prime K probibilità del rank
        #Controllo le camere
        q_cam=query_cams[i]
        candidates_cams=np.array(gallery_cams)[ranks_index[:,i][0:K]]

        q_exp=0 
        probability_sum=0
        for j in range(K):
            if q_cam != candidates_cams[j]:
                if AQE:
                    probability=1
                else:
                    probability=candidates_probability[j]
                if probability > soglia:
                    x=gallery[candidates_index[j]] #Dalla gallery prendo il feature vector del candidato
                    q_exp += probability*x
                    probability_sum += probability 
        q_exp=(q_exp +query[i])/(probability_sum +1)
        q_expansion[i]=q_exp
    return q_expansion










