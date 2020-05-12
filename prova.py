# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:13:17 2020

@author: AleCabiz
"""

# import os
# import glob
# from PIL import Image
# import numpy as np
# from scipy import linalg
# import matplotlib.pyplot as pl
# from importData import *
# from histogram import *
# from evaluation import *
# from Lbp import *
# from hog import *
# from BayesianModel import *
# import time
# import random
# import pickle

def calculateCmcFromRanks1(ranks,id_probes):
    rank=np.zeros((len(ranks),1))
    i=0
    for p_id in id_probes:
        j=0
        p_found=False
        if len(ranks.shape) == 1:
            rank_tmp=ranks
        else:
            rank_tmp=ranks[:,i]
        while p_found == False and j<len(rank_tmp):
            if p_id == rank_tmp[j]:
                p_found = True
            else:
                p_found = False
                j += 1
        rank[j] += 1
        i +=1
    cmc=np.cumsum(rank)/len(id_probes)
    return np.array(cmc)

def calculateCmcFromRanks2(ranks,id_probes):
    i=0
    cmc_vector=np.zeros(len(ranks))
    for y in q_id:
        rank=ranks[:,i]
        index=np.where(rank==y)[0][0] #restituisce l'indice della posizione più alta
        cmc_vector[index] += 1
        i += 1
    cmc_vector=np.cumsum(cmc_vector)/len(q_id) 
    return cmc_vector
    

r0,r1,r2,r3=ranks

cmc_vector2=calculateCmcFromRanks2(rr,q_id[0:10])

start=time.time()
for r in ranks:
    cmc_vector2=calculateCmcFromRanks2(r,q_id)
end=time.time()
print(end-start)


start=time.time()
for r in ranks:
    cmc_vector1=calculateCmcFromRanks1(r,q_id)
end=time.time()
print(end-start)


print((len(train_id)*(len(train_id)-1))/2)
print(len(B.d_sameId)+len(B.d_differentId))

print(len(B.d_sameId))
labels=list(set(train_id))
count=0
for y in labels:
    cnt=train_id.count(y)
    count += (cnt*(cnt-1))/2
print(count)

#Simulo degli errori nel feedback, quindi scambio di proposito gli id   
# labels=list(set(query_id))
# for y in labels:
#     count_istances=gallery_id.count(y)
#     #Simulo di scambiarne 1 su 10
#     for i in range(int(count_istances/10)):
#         y_index=gallery_id.index(y)
#         gallery_id[y_index]=np.random.choice(gallery_id)

ranks_wrongFeedback=wrongFeedback(ranks_label,query_id,15)
ranks_wfeed.append(ranks_wrongFeedback)
print('Ranks calcolato')

#Calcolo la cmc con wrong feedback 
cmc_vector_wfeed=calculateCmcFromRanks(ranks_wrongFeedback,query_id)
vettori_cmc_wfeed.append(cmc_vector_wfeed)

#Calcolo mAP con wrong feedback
mAP_wfeed=calculate_mAP(ranks_wrongFeedback,query_id,len(ranks_wrongFeedback))
mAP_list_wfeed.append(mAP_wfeed)

results_wfeed.append([k,n,ranks_wfeed,vettori_cmc_wfeed,mAP_list_wfeed])



# i=0
# for q in q_id:
#     rank=r0[:,i]
#     index=rank.index(q)
#     print(index)
#     i += 1

directory=[Dir1,Dir2,Dir3,Dir4,Dir5,Dir6,Dir7,Dir8,Dir9,Dir10]


Dir_cartella='..//Risultati test//'
names=[Dir.split('//')[2].split('.')[0] + '_ranks.pkl' for Dir in directory]
new_dir=[Dir_cartella + s for s in names]

for i in range(len(names)):
    f=open(directory[i],'rb')
    results=pickle.load(f)
    f.close()
    n_id,q_id,risultati=results
    X=[n_id,q_id]
    x=[]
    for r in risultati:
        x.append(r[0:3])
    X.append(x)
    f=open(new_dir[i],'wb')
    pickle.dump(X,f)
    f.close()         
        
        
         
