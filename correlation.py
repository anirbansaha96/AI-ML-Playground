import pandas as pd
import numoy as np
dataset=pd.read_csv('dataLarge',header=None,delimiter=' ',usecols=[2,3]);
dataset.corr(method ='pearson') 

#If we can't load the entire series onto the RAM at the same time, we may use chunking:
total=np.zeros((3,1));
totalsquare=np.zeros((3,1));
totalrows=0;
for chunk in pd.read_csv('dataLarge',chunksize=10000,delimiter=' ',index_col=0,header=None,usecols=[0,2,3]):
    dataset=chunk;
    dataset['X*Y']=dataset[2]*dataset[3];
    total+=pd.DataFrame(dataset.sum());
    totalsquare+=pd.DataFrame((dataset**2).sum());
    [nrows,ncols]=dataset.shape;
    totalrows+=nrows;
mean=total/totalrows;    
standard_deviation=((totalsquare/totalrows)-(mean**2))**0.5;
totalmultiplication=total.iat[2,0];
corr=(((totalmultiplication/totalrows)-(mean.iat[0,0]*mean.iat[1,0]))/(standard_deviation.iat[0,0]*standard_deviation.iat[1,0]))
corr
