from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import sys, os

def query_shift(query, gallery, positive, negative):#ROCCHIO

    if not positive: #if positive is empty
        new_query = np.expand_dims(query,axis=0) - 0.35*np.mean(gallery[negative, :],axis=0)
    else:
        if not negative:#if negative is empty
            new_query = np.expand_dims(query,axis=0) + 0.65*np.mean(gallery[positive, :],axis=0)
        else:
            new_query = np.expand_dims(query,axis=0) + 0.65*np.mean(gallery[positive, :],axis=0) - 0.35*np.mean(gallery[negative, :],axis=0)
    distances = euclidean_distances(new_query, gallery)
    rank = np.argsort(distances)

    return (distances, rank)   
