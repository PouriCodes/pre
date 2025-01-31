#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM
import streamlit as st
from PIL import Image
plt.style.use('fivethirtyeight')
import yfinance as yf
import pandas_datareader.data as wb


# In[27]:


st.write('''
# PRICE PREDICTOR

**mql-net.ir**
''')
img=Image.open('D:/hh.jpg')
st.image(img,width=600)


# In[28]:
symbol1 = 'BTC-USD'
symbol2 = 'ETH-USD'
symbol3 = 'ADA-USD'
symbol4 = 'SOL-USD'
# yf.pdr_override()
import datetime as dt


symbol1_ = yf.Ticker(symbol1)
df1 = symbol1_.history(start = "2020-01-01", end= "2025-10-10", period = '1d')
symbol2_ = yf.Ticker(symbol2)
df2 = symbol2_.history(start = "2020-01-01", end= "2025-10-10", period = '1d')
symbol3_ = yf.Ticker(symbol3)
df3 = symbol3_.history(start = "2020-01-01", end= "2025-10-10", period = '1d')
symbol4_ = yf.Ticker(symbol4)
df4 = symbol4_.history(start = "2020-01-01", end= "2025-10-10", period = '1d')
df1.to_csv('BTC.csv')
df2.to_csv('ETH.csv')
df3.to_csv('ADA.csv')
df4.to_csv('SOL.csv')
st.sidebar.header('INSERT DATA')
def data():
    symbol=st.sidebar.selectbox('select the symbol',['Bitcoin','Ethereum','Cardano','Solana'])
    return symbol


# In[29]:


def get_data(symbol):
    if symbol=='Bitcoin':
        df=pd.read_csv('D:/BTC.csv')
    elif symbol=='Ethereum':
        df=pd.read_csv('D:/ETH.csv')
    elif symbol=='Cardano':
        df=pd.read_csv('D:/ADA.csv')
    elif symbol=='Solana':
        df=pd.read_csv('D:/SOL.csv')
    df=df.set_index(pd.DatetimeIndex(df['Date'].values))
    return df


# In[30]:


symbol =data()
df=get_data(symbol)
data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*0.8)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)


# In[31]:


training_data=scaled_data[0:training_data_len , :]


# In[32]:


xtrain=[]
ytrain=[]
n=60


# In[33]:


for i in range(n,len(training_data)):
    xtrain.append(training_data[i-n:i , 0])
    ytrain.append(training_data[i,0])


# In[34]:


xtrain , ytrain = np.array(xtrain),np.array(ytrain)


# In[35]:


xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
#xtrain.shape


# In[36]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(xtrain.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[37]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=1,epochs=1)


# In[38]:


test_data=scaled_data[training_data_len - n : , :]
xtest=[]
ytest=dataset[training_data_len: , :]
for i in range(n,len(test_data)):
    xtest.append(test_data[i-n:i,0])


# In[39]:


xtest=np.array(xtest)
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))


# In[40]:


prediction=model.predict(xtest)
prediction=scaler.inverse_transform(prediction)
rmse =np.sqrt(np.mean(((prediction- ytest)**2)))
st.header('RMSE: ')
st.success(rmse)


# In[41]:


train=data[:training_data_len]
valid=data[training_data_len:]
valid['prediction']=prediction


# In[48]:



plt.figure(figsize=(12,4))
plt.title('PREDICTOR')
plt.xlabel('Date')
plt.ylabel('Price')
COLOR = 'blue'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.plot(train['Close'])
plt.plot(valid[['Close','prediction']])
plt.legend(['Train','Value','Prediction'])
plt.savefig('D:/accuracy.png', dpi=600)
plt.show()


# In[49]:


st.header('STOCK PREDICTOR ACCURACY : ')
imag=Image.open('D:/accuracy.png')
st.image(imag,width=600)


# In[50]:


newdf=data[-60:].values
#newdf


# In[51]:


snewdf=scaler.transform(newdf)


# In[52]:


xtest=[]
xtest.append(snewdf)
xtest=np.array(xtest)
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))


# In[53]:


pred=model.predict(xtest)
pred=scaler.inverse_transform(pred)
st.header('predicted price for next day:')
st.success(pred)
#pred


# In[ ]:





# In[ ]:




