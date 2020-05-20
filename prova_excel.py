# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:36:49 2020

@author: aleca
"""


import os
import pandas as pd
import openpyxl
import pickle
from openpyxl.utils.dataframe import dataframe_to_rows

Dir_fileDirectory='..//Risultati test//Directory_test.txt'
fileDirectory=open(Dir_fileDirectory,'r')
directory=[d.rstrip('\n') for d in fileDirectory.readlines()]

iterazioni,dataset,test_name,number_of_identity,k_values,rank1_values,mAP_values=[],[],[],[],[],[],[]
for Dir in directory:
    #Open file
    f=open(Dir,'rb')
    results=pickle.load(f)
    f.close()

    n_id,query_id,risultati=results
    
    for r in risultati:
        k,n,vettori_cmc,mAP=r
        #Rank1
        for v in vettori_cmc:
            rank1_values.append(v[0])
        #K
        for i in range(n+1):
            k_values.append(k)
        #Iterazioni
        for i in range(n+1):
            iterazioni.append(i)
        #mAP
        for i in mAP:
            mAP_values.append(i)
        #n_id
        for i in range(n+1):
            number_of_identity.append(n_id)
        #Dataset    
        for i in range(n+1):
            dataset.append(Dir.split('//')[2])
        #Test_name    
        for i in range(n+1):
            test_name.append(Dir.split('//')[-1].split('.')[0])

print(len(iterazioni))

wb = openpyxl.Workbook()
df=pd.DataFrame({'Iterazione':iterazioni,'Dataset':dataset,'Test name':test_name,
                 'Numero Identità':number_of_identity,'K':k_values,'rank1':rank1_values,'mAP':mAP_values})

print('DATAFRAME CREATO')    
    
ws = wb.active 


for r in dataframe_to_rows(df,index=False):
    ws.append(r)
    
wb.save('..//Risultati test//Risultati_Test.xlsx')
