# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:11:25 2020

@author: aleca
"""


import os
import pandas as pd
import numpy as np
import openpyxl
import pickle
from openpyxl.utils.dataframe import dataframe_to_rows


#Duke

Dir1='..//Risultati test//Duke//Duke_300pics_k_n_withoutFeedback_AQE.pkl'
Dir2='..//Risultati test//Duke//Duke_300pics_k_n_withoutFeedback_soglia0,5.pkl'
Dir3='..//Risultati test//Duke//Duke_test_complete_AQE.pkl'
Dir4='..//Risultati test//Duke//Duke_test_complete_AQE_Similarity.pkl'
Dir5='..//Risultati test//Duke//Duke_test_complete_soglia0,5.pkl'    
Dir6='..//Risultati test//Duke//Duke_test_complete_Similarity.pkl'
    
#    Feedback   
     
Dir7='..//Risultati test//Duke//Duke_300pics_k_n_withFeedback_Prob.pkl'
Dir8='..//Risultati test//Duke//Duke_300pics_k_n_withFeedback_Prob1.pkl'
Dir9='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55.pkl'
Dir10='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'  
Dir11='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob1_k55.pkl'   
Dir12='..//Risultati test//Duke//Duke_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'    

    
#############################################
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

#CrossDataset
Dir25='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_AQE.pkl'
Dir26='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_soglia0,5.pkl'
Dir27='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_HumanFeedback_Prob_k55.pkl'
Dir28='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_HumanFeedback_Prob1_k55.pkl'

Dir29='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_AQE_Similarity.pkl'
Dir30='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_soglia0,5_Similarity.pkl'
Dir31='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'
Dir32='..//Risultati test//MarketFromDuke//MarketFromDuke_test_complete_HumanFeedback_Prob1_k55_Similarity.pkl'

#Duke
Dir33='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_AQE.pkl'
Dir34='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_soglia0,5.pkl'
Dir35='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_HumanFeedback_Prob_k55.pkl'
Dir36='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_HumanFeedback_Prob1_k55.pkl'

Dir37='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_AQE_Similarity.pkl'
Dir38='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_soglia0,5_Similarity.pkl'
Dir39='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'
Dir40='..//Risultati test//DukeFromMarket//DukeFromMarket_test_complete_HumanFeedback_Prob1_k55_Similarity.pkl'


directory_duke=[Dir3,Dir5,Dir9,Dir11]
directory_duke_similarity=[Dir4,Dir6,Dir10,Dir12]
directory_DukeFromMarket=[Dir33,Dir34,Dir35,Dir36]
directory_DukeFromMarket_Similarity=[Dir37,Dir38,Dir39,Dir40]


directory_market=[Dir16,Dir18,Dir21,Dir23]
directory_market_similarity=[Dir17,Dir15,Dir22,Dir24]

directory_MarketFromDuke=[Dir25,Dir26,Dir27,Dir28]
directory_MarketFromDuke_Similarity=[Dir29,Dir30,Dir31,Dir32]


#directory_300Duke=[Dir1,Dir2,Dir7,Dir8]
#directory_300Market=[Dir13,Dir14,Dir19,Dir20]

rows_baseline=[]
rows=[]
for Dir in directory_MarketFromDuke_Similarity:
    #Open file
    f=open(Dir,'rb')
    results=pickle.load(f)
    f.close()

    n_id,query_id,risultati=results


    for r in risultati:
        k,n,vettori_cmc,vettore_mAP=r
        vettori_cmc=[v.round(4)*100 for v in vettori_cmc]
        vettore_mAP=[round(i,4)*100 for i in vettore_mAP]
        for i in np.arange(1,n+1):
            riga=[]
            riga.append(Dir.split('//')[-1])
            riga.append(i)
            riga.append(vettore_mAP[i])
            riga.append(vettori_cmc[i][0])
            riga.append(vettori_cmc[i][4])
            riga.append(vettori_cmc[i][9])
            riga.append(vettori_cmc[i][19])
            rows.append(riga)
            #Baseline
            rows_baseline.append(['Baseline',None,vettore_mAP[0],vettori_cmc[0][0],
                               vettori_cmc[0][4],vettori_cmc[0][9],vettori_cmc[0][19]])
        
#Controllo baseline
baseline_check=False
for i in range(len(rows_baseline)-1):
    if rows_baseline[0] != rows_baseline[i+1]:
        print('ERRORE NELLE BASELINE')
        baseline_check=False
    else:
        baseline_check=True
if baseline_check:
    baseline_row=rows_baseline[0]
    
    
wb = openpyxl.Workbook()
colonne=['Metodi','Iterazione','mAP','rank1','rank5','rank10','rank20']
df=pd.DataFrame([baseline_row],columns=colonne)
df1=pd.DataFrame(rows,columns=colonne)
df=df.append(df1,ignore_index = True)


print('DATAFRAME CREATO')    
    
ws = wb.active 
#ws = wb.create_sheet("Duke")

for r in dataframe_to_rows(df,index=False):
    ws.append(r)
    
wb.save('..//Risultati test//tab.xlsx')