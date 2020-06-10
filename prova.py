# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:14:11 2020

@author: aleca
"""


from BayesianModel import *
import pickle

##Market  
#
#Dir13='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_AQE.pkl'
#Dir14='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_soglia0,5.pkl' 
#Dir15='..//Risultati test//Market//Market_test_complete_Similarity.pkl'    
#Dir16='..//Risultati test//Market//Market_test_complete_AQE.pkl'
#Dir17='..//Risultati test//Market//Market_test_complete_AQE_Similarity.pkl'
#Dir18='..//Risultati test//Market//Market_test_complete_soglia0,5.pkl'    
#
##    Feedback   
#
#Dir19='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob.pkl'
#Dir20='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob1.pkl' 
#Dir21='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55.pkl'
#Dir22='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'    
#Dir23='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob1_k55.pkl'
#Dir24='..//Risultati test//Market//Market_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'
#
#nuova_cartella='C://Users//aleca//Desktop//Nuova cartella//Market//Ranks-'
#
##directory_market=[Dir16,Dir18,Dir21,Dir23]
##directory_market_similarity=[Dir17,Dir15,Dir22,Dir24]
##
##directory_market2=[nuova_cartella +d.split('//')[-1] for d in directory_market]
#dir_ranks='..//Risultati test//Market//Ranks//Ranks-'
##directory_market3=[dir_ranks +d.split('//')[-1] for d in directory_market_similarity]
#
#dir_prova=[nuova_cartella +Dir16.split('//')[-1],dir_ranks +Dir17.split('//')[-1]]
def evaluation_forEachIdentity(ranks,id_query):
    id_query=np.array(id_query)
    labels=list(set(id_query))
    r1_list,mAP_list=[],[]
    for q in labels:
        colonne=np.array(id_query==q)
        rank_tmp=ranks[:,colonne]
        rank1=calculateCmcFromRanks(rank_tmp,[q])[0] #Calcolo rank1
        mAP=(calculate_mAP(rank_tmp,[q],len(rank_tmp)))
        r1_list.append(rank1)
        mAP_list.append(mAP)
    print('rank1 mean' + str(np.mean(r1_list)))
    print('map mean' + str(np.mean(mAP_list)))

    return [r1_list,mAP_list,labels]

d1='..//Risultati test//RanksMarket//Ranks-Market_test_complete_AQE_Similarity.pkl'
d2='..//Risultati test//RanksMarket//Ranks-Market_test_complete_AQE.pkl'
d3='..//Risultati test//Ranks-DukeFromMarket//Ranks-DukeFromMarket_test_complete_AQE_Similarity.pkl'
d4='..//Risultati test//Ranks-DukeFromMarket//Ranks-DukeFromMarket_test_complete_AQE.pkl'

dd=[d1,d2]
X,q=[],[]
#Open file
for d in dd:
    f=open(d,'rb')
    results=pickle.load(f)
    f.close()
    
    n_id,query_id,risultati=results
    x=[]
    for r in risultati:
        k,n,ranks=r
#        somma=np.sum(ranks[0][0,:]==query_id)
#        index_false=np.where(ranks[0][0,:]!=query_id)
#        print(calculateCmcFromRanks(ranks[0],query_id)[0])
#        print(calculate_mAP(ranks[0],query_id,len(ranks[0])))
#        l=evaluation_forEachIdentity(ranks[0],query_id)
#        x.append(index_false)
#        x.append(somma)
#        print(len(query_id))
#        print(somma)
        X.append(ranks[0])
        q.append(query_id)
         
rs,rp=X
print(q[0]==q[1])
print(np.sum(rp[0]==q[0]))
q_id=q[0]

rank1p=calculateCmcFromRanks(rp,q_id)[0]
rank1s=calculateCmcFromRanks(rs,q_id)[0]
mAPp=calculate_mAP(rp,q_id,len(rp))
mAPs=calculate_mAP(rs,q_id,len(rp))

print((rank1p,rank1s))
print((mAPp,mAPs))


        
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
