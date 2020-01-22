# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:47:48 2020

@author: anirb
"""
import numpy as np
import pandas as pd
total=np.zeros((25,1));
totalrows=0;
for chunk in pd.read_csv('dataLarge',chunksize=10000,delimiter=' ',index_col=0,header=None):
    dataset=chunk;
    total+=pd.DataFrame(dataset.sum());
    [nrows,ncols]=dataset.shape;
    totalrows+=nrows;
mean=total/totalrows;    
    