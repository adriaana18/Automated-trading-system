# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:53:42 2021

@author: adria
"""

"""
Created on Fri May  7 17:20:16 2021

@author: adria
"""
#%%
"""
CODE PREP
"""
#%%
"""
Create & set directory
"""
import os
os.chdir("C:/Users/adria/OneDrive/Documents/Atm/EC331_RAE/Code")

#%%
"""
Import libraries
"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
#import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
#%%
"""
DATA PREPROCESSING
"""
#%%
"""
Load & store the data
"""
AAPL = pd.read_csv('AAPL_bloomberg.csv')
#Show the data
print(AAPL)
#%%
"""
Data exploration
"""
#Sate date column to date
AAPL['Date'] =pd.to_datetime(AAPL.Date, dayfirst='true')
# Sort DataFrame by date
AAPL = AAPL.sort_values('Date')
print(AAPL['Date'])
# Double check the result
AAPL.head()
print(AAPL)

#%%
"""
Plot the data: data visualisation
"""
plt.figure(figsize = (30,15))
plt.plot(range(AAPL.shape[0]),(AAPL['PX_LOW']+AAPL['PX_HIGH'])/2.0, label = 'AAPL')
plt.xticks(range(0,AAPL.shape[0],500),AAPL['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Mid Price',fontsize=20)
plt.legend(loc='upper right')
plt.show()

#%%
"""
Split data into train and test sets
"""
# Calculate the mid prices from the highest and lowest prices of the day
high_prices = AAPL.loc[:,'PX_HIGH'].to_numpy()
low_prices = AAPL.loc[:,'PX_LOW'].to_numpy()
mid_prices = (high_prices+low_prices)/2.0
#Split data: rule of thumb 80% train data, 20% test data
train_data = mid_prices[:5300]
test_data = mid_prices[5300:]
#%%
"""
Normalize the data
"""
# Scale the data to be between 0 and 1
# Normalize both test and train data with respect to training data 
        #because you are not supposed to have access to test data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
# Train the Scaler with training data and smooth data
smoothing_window_size = 1000
for di in range(0,4500,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit (last window) of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# Reshape both train and test data
train_data = train_data.reshape(-1)
    # Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(5300):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Concatenate normalized data - Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)

#%%%
#%%%
"""
ONE-STEP AHEAD MOVING AVERAGE PREDICTION
"""
#%%
"""
Standard MA
"""
window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    if pred_idx >= N:
        date = dt.datetime.strptime('%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = AAPL.loc[pred_idx,'Date']
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
#0.01123

#Plot predictions against true values
plt.figure(figsize = (18,9))
plt.plot(range(AAPL.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()
#%%
"""
Exponential MA
"""
window_size = 100
N = train_data.size
run_avg_predictions = []
run_avg_x = []
mse_errors = []
running_mean = 0.0
run_avg_predictions.append(running_mean)
decay = 0.5

for pred_idx in range(1,N):
    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))
#0.00009

#Plot predictions against true values
plt.figure(figsize = (18,9))
plt.plot(range(AAPL.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()
