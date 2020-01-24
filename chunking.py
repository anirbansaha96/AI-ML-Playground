#We use chunking to calculate the mean and standard deviation of a 4999998*25 size data.
import numpy as np
import pandas as pd
total=np.zeros((25,1));
totalsquare=np.zeros((25,1));
totalrows=0;
for chunk in pd.read_csv('dataLarge',chunksize=10000,delimiter=' ',index_col=0,header=None):
    dataset=chunk;
    total+=pd.DataFrame(dataset.sum());
    totalsquare+=pd.DataFrame((dataset**2).sum());
    [nrows,ncols]=dataset.shape;
    totalrows+=nrows;
mean=total/totalrows;    
standard_deviation=((totalsquare/totalrows)-(mean**2))**0.5;
