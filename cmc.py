# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:34:59 2020

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

#Restituisce maching_position che contiene la posizione del id_a-iesima nel risulatato del matching
#e rank_vector, che contiene il conteggio di quante volte id-esimo si trova nella posizione i-esima.
def CMC_curve(camA,camB): 
    similarity_dict={}
    similarity_dict=similarity_dictionary(camA,camB)
    similarity_dict_sort={}
    for k in similarity_dict.keys():
        sort_dict=sortDictForValue(similarity_dict[k])
        #sort_dict=sorted(similarity_dict[k].items(), key= lambda kv:(kv[1],kv[0]),reverse=True)
        similarity_dict_sort[k] = sort_dict
    #maching_position contiene la posizione del id_a-iesima nel risulatato del matching
    maching_position=[]
    for id_a in similarity_dict_sort.keys():
        id_a_found=False
        match_result=similarity_dict_sort[id_a]
        r=1
        while(id_a_found==False):
            if id_a != match_result[r-1][0]:
                r += 1
            else:
                id_a_found = True
        maching_position.append(r) 
        
    #Calcolo il quante volte id_a è stato trovato dentro un certo rank
    rank_vector=np.zeros((1,len(camB.keys())))
    for i in range(max(maching_position)):
            #rank_vector.append(maching_position.count(i+1))
            rank_vector[0,i]=maching_position.count(i+1)
    cmc_vector=np.cumsum(np.array(rank_vector))/len(camB.keys())
    return maching_position,cmc_vector         
            
#similarity_dict è un dizionario le cui chiavi sono gli id_a e i valori un altro dizionario
#con chiave  id_b e valore il rispettivo grado di similarità con id_a
def similarity_dictionary(camA,camB): 
    Id_A=camA.keys()
    Id_B=camB.keys()
    similarity_dict={}
    K={}
    for i in Id_A:
        A=camA[i]
        a_r,a_g,a_b=histogram_vector(A)
        for j in Id_B:
            B=camB[j]
            b_r,b_g,b_b=histogram_vector(B)
            
            kr=histogram_intersection(a_r,b_r) 
            kg=histogram_intersection(a_g,b_g) 
            kb=histogram_intersection(a_b,b_b) 
            k=(kr+kg+kb)/3 
            K[j]=k
        similarity_dict[i]=K
    return similarity_dict
  
#Ordino i dizionari interni in ordine decrescente di similarità      
def sortDictForValue(d):
    sort_dict=sorted(d.items(), key= lambda kv:(kv[1],kv[0]),reverse=True)
    return sort_dict
    

#Plot CMC
def plot_CMC(rank_vector,Id_A):
    x=np.arange(len(Id_A))+1
    y=np.cumsum(np.array(rank_vector))/len(Id_A) #Faccio la somma cumulativa e dividento per il numero totale ottengo le probabilità        
    pl.plot(x,y)
    pl.show()        

if __name__ == '__main__':
    
    da={k: v for k,v in camA.items() if k<4}
    db={k: v for k,v in camB.items() if k<5}
    Id_A=da.keys()
    maching_position,rank_vector=CMC_curve(da,db)
    plot_CMC(rank_vector,Id_A)


