import pandas as pd
import numpy as np
dataset=pd.read_csv('trades.txt',delimiter= ' ');

#Initialising Variables to be used throughout the program
position=0;
averageopenprice=0;
pnlunrealized=0;
pnlrealized=0;
pnltotal=0;

#Inserting Columns into the dataset to visualise the working in a better way
dataset.insert(2,'Position',np.zeros(len(dataset.index)));
dataset.insert(3,'Average Open Price',np.zeros(len(dataset.index)));
dataset.insert(4,'PnL Unrealized',np.zeros(len(dataset.index)));
dataset.insert(5,'PnL Realized',np.zeros(len(dataset.index)));
dataset.insert(6,'PnL Total',np.zeros(len(dataset.index)));

#Algorithm Begins
for i in range(0,len(dataset.index)):
    qty=dataset.iat[i,0]; #This will help in writing more legible codes
    price=dataset.iat[i,1];
    if position==0:
        position=qty;
        averageopenprice=price;
        pnlrealized=pnlrealized;
        pnlunrealized=0;
        pnltotal=pnlrealized+pnlunrealized;
    elif position>0:
        if qty>0:
            averageopenprice=(position*averageopenprice+qty*price)/(position+qty);
            position+=qty;
            pnlrealized=pnlrealized;
            pnlunrealized=(price-averageopenprice)*position;
            pnltotal=pnlrealized+pnlunrealized;
        elif qty<0:
            if abs(qty)<position:
                averageopenprice=averageopenprice;
                pnlrealized+=(price-averageopenprice)*abs(qty);
                position+=qty;
                pnlunrealized=(price-averageopenprice)*position;
                pnltotal=pnlrealized+pnlunrealized;
            elif abs(qty)==position:
                pnlrealized+=(price-averageopenprice)*abs(qty);
                position=0;
                averageopenprice=0;
                pnlunrealized=0;
                pnltotal=pnlrealized+pnlunrealized;
            elif abs(qty)>position:
                pnlrealized+=(price-averageopenprice)*position;
                position+=qty;
                averageopenprice=price;
                pnlunrealized=(price-averageopenprice)*position;
                pnltotal=pnlrealized+pnlunrealized;
    elif position<0:
        if qty<0:
            averageopenprice=(abs(position)*averageopenprice+abs(qty)*price)/(abs(position+qty));
            position+=qty;
            pnlrealized=pnlrealized;
            pnlunrealized=(averageopenprice-price)*abs(position);
            pnltotal=pnlrealized+pnlunrealized;
        elif qty>0:
            if qty<abs(position):
                averageopenprice=averageopenprice;
                pnlrealized=(averageopenprice-price)*qty;
                position+=qty;
                pnlunrealized=(averageopenprice-price)*abs(position);
                pnltotal=pnlrealized+pnlunrealized;
            elif qty==abs(position):
                pnlrealized+=(averageopenprice-price)*qty;
                position=0;
                averageopenprice=0;
                pnlunrealized=0;
                pnltotal=pnlrealized+pnlunrealized;
            elif qty>abs(position):
                pnlrealized+=(averageopenprice-price)*abs(position);
                position+=qty;
                averageopenprice=price;
                pnlunrealized=0;
                pnltotal=pnlrealized+pnlunrealized;
    dataset.iat[i,2]=position;
    dataset.iat[i,3]=averageopenprice;
    dataset.iat[i,4]=pnlunrealized;
    dataset.iat[i,5]=pnlrealized;
    dataset.iat[i,6]=pnltotal;


output=dataset[['qty','price','PnL Realized','PnL Unrealized','PnL Total']];    