'''This is a simple program to find the number of null values in a dataset,
 when the size of the dataset is too Large in this case 4999999*25, 
 it becomes really difficult to load such a dataset onto the RAM, 
 so we use indexing and reduce the workload on the RAM'''
 
import numpy as np
import pandas as pd
import numpy as np
null_cols=np.ones((25,1))
for i in range(1,25):
    dataset=pd.read_csv('dataLarge',delimiter=' ',index_col=0,header=None,usecols=[0,i]);
    null_cols[i-1]=dataset[i].isnull().sum()