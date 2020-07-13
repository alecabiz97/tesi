# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:20:36 2020

@author: aleca
"""


import os
import pandas as pd
import numpy as np
from importData import *

import openpyxl
import pickle
from openpyxl.utils.dataframe import dataframe_to_rows

Dir1='..//Risultati test//DukeProbability'
Dir2='..//Risultati test//DukeSimilarity'
Dir3='..//Risultati test//MarketProbability'
Dir4='..//Risultati test//MarketSimilarity'

Dir5='..//Risultati test//DukeFromMarketProbability'
Dir6='..//Risultati test//DukeFromMarketSimilarity'
Dir7='..//Risultati test//MarketFromDukeProbability'
Dir8='..//Risultati test//MarketFromDukeSimilarity'

Dir9='..//Risultati test//DukeFeedback'
Dir10='..//Risultati test//MarketFeedback'
Dir11='..//Risultati test//DukeFromMarketFeedback'
Dir12='..//Risultati test//MarketFromDukeFeedback'

Dir13='..//Risultati test//DukeCoseno'
Dir14='..//Risultati test//MarketCoseno'
Dir15='..//Risultati test//DukeFromMarketCoseno'
Dir16='..//Risultati test//MarketFromDukeCoseno'

Dir17='..//Risultati test//Duke_CameraFilter'
Dir18='..//Risultati test//Market_CameraFilter'
Dir19='..//Risultati test//DukeFromMarket_CameraFilter'
Dir20='..//Risultati test//MarketFromDuke_CameraFilter'


n,X=loadFiles(Dir18)

rows_baseline=[]
rows=[]
for results,filename in zip(X,n):

    n_id,query_id,risultati=results

    for r in risultati:
        k,n,vettori_cmc,vettore_mAP=r
        vettori_cmc=[v.round(4)*100 for v in vettori_cmc]
        vettore_mAP=[round(i,4)*100 for i in vettore_mAP]
        for i in np.arange(1,n+1):
            riga=[]
            riga.append(filename)
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
    
wb.save('..//Risultati test//tab2.xlsx')