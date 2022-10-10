# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:34:02 2021

@author: adria
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('presictions_test.csv')

#function that indicates buy/sell signal
def buy_sell(df):
    Buy = []
    Sell = []
    flag = -1
    
    for i in range(0, len(df.index)):
        if df['Return_predictions'][i]> 0:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(df['Return_predictions'][i])
                flag = 1 
            else:
                Buy.append(np.nan)
        elif df['Return_predictions'][i] < 0:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(df['Return_predictions'][i])
                flag = 0
                
            else:
                Sell.append(np.nan)
            
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
                
    return(Buy, Sell)

#Use the function to createa buy/sell signal
a = buy_sell(df)
df['Buy_Signal_Price'] = a[0]
df['Sell_Signal_Price'] = a[1]

#Show buy/sell signals
plt.figure(figsize=(12,6))
plt.scatter(df.index, df['Buy_Signal_Price'],
            color='green', 
            label='Buy', 
            marker = '^', 
            alpha=1)
plt.scatter(df.index, df['Sell_Signal_Price'],
            color='red', 
            label='Sell', 
            marker = 'v', 
            alpha=1)
plt.plot(df['Return_predictions'], label='Predicted retunrs', alpha = 0.9)
plt.title('Predicted returns; Buy & Sell signals')
plt.xticks(rotation = 45)
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend(loc='upper left')
plt.show()