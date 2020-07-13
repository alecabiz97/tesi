# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:47:41 2020

@author: aleca
"""

import os
import numpy as np
from utils.importData import *

import pickle

Dir1='..//Risultati test//DukeProbability'
Dir2='..//Risultati test//DukeSimilarity'
Dir3='..//Risultati test//MarketProbability'
Dir4='..//Risultati test//MarketSimilarity'

Dir5='..//Risultati test//DukeFromMarketProbability'
Dir6='..//Risultati test//DukeFromMarketSimilarity'
Dir7='..//Risultati test//MarketFromDukeProbability'
Dir8='..//Risultati test//MarketFromDukeSimilarity'


Dir9='..//Risultati test//DukeCoseno'
Dir10='..//Risultati test//MarketCoseno'
Dir11='..//Risultati test//DukeFromMarketCoseno'
Dir12='..//Risultati test//MarketFromDukeCoseno'

directory=[Dir1,Dir2,Dir3,Dir4,Dir5,Dir6,Dir7,Dir8,Dir9,Dir10,Dir11,Dir12]
for Dir in directory:
    title=Dir.split('//')[-1]
    n,X=loadFiles(Dir)
    values_rank1,values_mAP=[],[]
    for results,filename in zip(X,n):
    
        n_id,query_id,risultati=results
    
        for r in risultati:
            k,n,vettori_cmc,vettore_mAP=r
            values_rank1.append([round(v[0],4)*100 for v in vettori_cmc])
            values_mAP.append([round(i,4)*100 for i in vettore_mAP])
    
    
    
    testname=['AQE','BQE','Feedback pesato','Feedback non pesato']
    x=range(0,4)
    #Plot grafico rank1
    for r1,metodo in zip(values_rank1,testname):
        pl.plot(x,np.array(r1),label=metodo)
    
    pl.ylim(70,100)
    pl.xticks(range(0,4))
    pl.legend()
    pl.grid(True)
    pl.ylabel('rank1(%)')
    pl.xlabel('n')
    pl.title(title)
    pl.savefig('..//Grafici_3Round//' + title + '_rank1.png')
    pl.show()
    
    #Plot grafico mAP
    for mAP,metodo in zip(values_mAP,testname):
        pl.plot(x,np.array(mAP),label=metodo)
    
    pl.ylim(70,100)
    pl.xticks(range(0,4))
    pl.legend()
    pl.grid(True)
    pl.ylabel('mAP(%)')
    pl.xlabel('n')
    pl.title(title)
    pl.savefig('..//Grafici_3Round//' + title + '_mAP.png')
    pl.show()











