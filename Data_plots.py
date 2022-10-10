# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 10:46:27 2021

@author: adria
"""
#%%
"""
Create & set directory
"""
import os
os.chdir("C:/Users/adria/OneDrive/Documents/RAE/Code")

#%%
"""
Import libraries
"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import math
#%%
"""
DATA PREPROCESSING
"""
#%%
"""
Load & store the data
"""
df = pd.read_csv('BA_bloomberg.csv')
#Show the data
print(df)
#%%
"""
Data exploration
"""
#Sate date column to date
df['Date'] = pd.to_datetime(df.Date, dayfirst='true')
for i in range (0, len(df['Date'])):
    df['Date'][i]=pd.datetime.strptime(str(df['Date'][i]), '%Y-%m-%d %H:%M:%S').date()
print (df['Date'][0])

# Sort DataFrame by date
df = df.sort_values('Date')
print(df['Date'])
# Double check the result
df.head()
print(df)

#%%
"""
Plot the data: data visualisation
"""

plt.figure(figsize = (30,15))
plt.plot(range(df.shape[0]),(df['PX_LOW']+df['PX_HIGH'])/2.0, label = 'Stock price', color='b')
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500], fontsize=20)
plt.yticks(df['PX_HIGH'].loc[::500], fontsize=20)
plt.title('Evolution of Boeing Stock Price Over Time',fontsize=40)
plt.xlabel('Date',fontsize=25)
plt.ylabel('Daily Mid Price',fontsize=25)
plt.legend(loc='upper right')
plt.show()



