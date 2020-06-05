# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:14:11 2020

@author: aleca
"""


from BayesianModel import *
import pickle

#Market  

Dir13='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_AQE.pkl'
Dir14='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_soglia0,5.pkl' 
Dir15='..//Risultati test//Market//Market_test_complete_Similarity.pkl'    
Dir16='..//Risultati test//Market//Market_test_complete_AQE.pkl'
Dir17='..//Risultati test//Market//Market_test_complete_AQE_Similarity.pkl'
Dir18='..//Risultati test//Market//Market_test_complete_soglia0,5.pkl'    

#    Feedback   

Dir19='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob.pkl'
Dir20='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob1.pkl' 
Dir21='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55.pkl'
Dir22='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'    
Dir23='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob1_k55.pkl'
Dir24='..//Risultati test//Market//Market_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'

nuova_cartella='C://Users//aleca//Desktop//Nuova cartella//Market//Ranks-'

#directory_market=[Dir16,Dir18,Dir21,Dir23]
#directory_market_similarity=[Dir17,Dir15,Dir22,Dir24]
#
#directory_market2=[nuova_cartella +d.split('//')[-1] for d in directory_market]
dir_ranks='..//Risultati test//Market//Ranks//Ranks-'
#directory_market3=[dir_ranks +d.split('//')[-1] for d in directory_market_similarity]

dir_prova=[nuova_cartella +Dir16.split('//')[-1],dir_ranks +Dir17.split('//')[-1]]

X=[]
for Dir in dir_prova:
    #Open file
    f=open(Dir,'rb')
    results=pickle.load(f)
    f.close()

    n_id,query_id,risultati=results
    x=[]
    x.append(Dir.split('//')[-1])
    for r in risultati:
        k,n,ranks=r
        somma=np.sum(ranks[0][,:]==query_id)
        index_false=np.where(ranks[0][0,:]!=query_id)
        x.append(index_false)
        x.append(somma)
        print(somma)
    X.append(x)
    
    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
