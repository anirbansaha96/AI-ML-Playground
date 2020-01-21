import pandas as pd
import numpy as np
dataset=pd.read_csv('trades.txt', delimiter=' ')

dataset.insert(2,"PnL",np.zeros(len(dataset.index)));
cumulative_amount=0;
[m,n]=dataset.shape;
for i in range(0,m):
    cumulative_amount+=-dataset.iat[i,0]*dataset.iat[i,1];
    dataset.iat[i,2]=cumulative_amount;
