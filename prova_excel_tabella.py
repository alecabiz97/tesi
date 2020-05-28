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
#    Dir='..//Risultati test//Duke//Duke_test_complete.pkl'

Dir5='..//Risultati test//Duke//Duke_300pics_k_n_withoutFeedback_AQE.pkl'
Dir6='..//Risultati test//Duke//Duke_300pics_k_n_withoutFeedback_soglia0,5.pkl'
Dir7='..//Risultati test//Duke//Duke_test_complete_AQE.pkl'
Dir8='..//Risultati test//Duke//Duke_test_complete_AQE_Similarity.pkl'
Dir9='..//Risultati test//Duke//Duke_test_complete_soglia0,5.pkl'    
Dir10='..//Risultati test//Duke//Duke_test_complete_Similarity.pkl'
    
#    Feedback   
     
Dir11='..//Risultati test//Duke//Duke_300pics_k_n_withFeedback_Prob.pkl'
Dir12='..//Risultati test//Duke//Duke_300pics_k_n_withFeedback_Prob1.pkl'
Dir13='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55.pkl'
Dir14='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'  
Dir15='..//Risultati test//Duke//Duke_test_complete_HumanFeedback_Prob1_k55.pkl'   
Dir16='..//Risultati test//Duke//Duke_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'    

    
#############################################
#Market  

Dir17='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_AQE.pkl'
Dir18='..//Risultati test//Market//Market_300pics_k_n_withoutFeedback_soglia0,5.pkl' 
Dir19='..//Risultati test//Market//Market_test_complete.pkl'
Dir20='..//Risultati test//Market//Market_test_complete_Similarity.pkl'    
Dir21='..//Risultati test//Market//Market_test_complete_AQE.pkl'
Dir22='..//Risultati test//Market//Market_test_complete_AQE_Similarity.pkl'
Dir23='..//Risultati test//Market//Market_test_complete_soglia0,5.pkl'    

#    Feedback   

Dir24='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob.pkl'
Dir25='..//Risultati test//Market//Market_300pics_k_n_withFeedback_Prob1.pkl' 
Dir26='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55.pkl'
Dir27='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob_k55_Similarity.pkl'    
Dir28='..//Risultati test//Market//Market_test_complete_HumanFeedback_Prob1_k55.pkl'
Dir29='..//Risultati test//Market//Market_test_complete_Similarity_HumanFeedback_Prob1_k55.pkl'

directory_duke=[Dir7,Dir9,Dir13,Dir15]
directory_duke_similarity=[Dir8,Dir10,Dir14,Dir16]

directory_market=[Dir21,Dir23,Dir26,Dir28]
directory_market_similarity=[Dir22,Dir20,Dir27,Dir29]


directory_300Duke=[Dir5,Dir6,Dir11,Dir12]
directory_300Market=[Dir17,Dir18,Dir24,Dir25]

rows_baseline=[]
rows=[]
for Dir in directory_300Market:
    #Open file
    f=open(Dir,'rb')
    results=pickle.load(f)
    f.close()

    n_id,query_id,risultati=results


    for r in risultati:
        k,n,vettori_cmc,vettore_mAP=r
        if ('withFeedback' in Dir and k==55) or ('withoutFeedback' in Dir and k==5):
            vettori_cmc=[v.round(3)*100 for v in vettori_cmc]
            vettore_mAP=[round(i,3)*100 for i in vettore_mAP]
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
    
wb.save('..//Risultati test//tab_market2.xlsx')