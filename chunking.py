#We use chunking to calculate the mean of a 4999998*25 size data.
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
