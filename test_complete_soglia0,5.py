# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:06:53 2020

@author: aleca
"""

print('START')

    
DirMarket = '..\\FeatureCNN\\Market-1501'
DirDuke = '..\\FeatureCNN\\DukeMTMC'

#Feature CNN
testData,queryData,trainingData=loadCNN(DirDuke)

#istogrammi RGB
#testData,queryData,trainingData=loadMarket_1501(feature=True)

test_cams, test_feature, test_ids, test_desc = testData
query_cams, query_feature, query_ids, query_desc = queryData
train_cams, train_feature, train_ids, train_desc = trainingData
    
print(len(test_ids))
print(len(query_ids))
print(len(train_ids))

#Load BayesianModel gia addestrato
#Bayes=loadFile('..\\Bayes_Market_trained.pkl')
Bayes=loadFile('..\\Bayes_Duke_trained.pkl')


gallery,gallery_id=test_feature,test_ids
query,query_id = query_feature[0::], query_ids[0::]

        
start=time.time()


print('START TEST')

n,k=3,5
vettori_cmc,ranks,mAP_list=[],[],[] 
for i in range(n+1):
    ranks_index,ranks_probability,ranks_label =calculateRanks(query,gallery,gallery_id,Bayes)
    ranks.append(ranks_label)
    print('Ranks calcolato')
    
    #Calcolo la cmc
    cmc_vector=calculateCmcFromRanks(ranks_label,query_id)
    vettori_cmc.append(cmc_vector)

    #Calcolo mAP
    mAP=calculate_mAP(ranks_label,query_id,len(ranks_label))
    mAP_list.append(mAP)
    
    
    #Calcolo la nuova query    
    query=queryExpansion(ranks_index,ranks_probability,gallery,query,k,AQE=False,soglia=0.5)
    print('Nuova query calcolata')
    

#Cmc e mAP
results=[len(set(query_id)),query_id]
k_n_cmc_mAP=[k,n,vettori_cmc,mAP_list]
results.append([k_n_cmc_mAP])

#Solo i ranks
results_ranks=[len(set(query_id)),query_id]
k_n_ranks=[k,n,ranks]
results_ranks.append([k_n_ranks])

rank1_mAP_functionOfn(results[2])

    
f=open('..//Risultati test//Duke_test_complete_soglia0,5.pkl','wb') 
pickle.dump(results,f)
f.close()

f=open('..//Risultati test//Ranks-Duke_test_complete_soglia0,5.pkl','wb') 
pickle.dump(results_ranks,f)
f.close()
     
    
end=time.time()
tempo=end-start
print(tempo)
