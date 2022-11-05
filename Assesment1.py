#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


path=('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv')
data=pd.read_csv(path)
data


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.head(52)


# In[7]:


data.info()


# In[8]:


data.isnull()


# In[9]:


data.isnull().sum()


# In[10]:


data.fillna(data.mean())


# In[11]:


data.isnull().sum()


# In[12]:


data = data.dropna()


# In[13]:


data.isnull().sum()


# In[14]:


data.info()


# In[31]:


x=data[['wheel-base','wheel-base', 'length', 'width', 'height','bore','stroke','horsepower','peak-rpm']].values

y=data[['price']].values


# In[32]:


x


# In[17]:


import sklearn


# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3,random_state=42)


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)


# In[21]:


pred=logmodel.predict(x_test)


# In[22]:


pred


# In[23]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,pred)


# In[24]:


accuracy_score(y_test,pred)


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


reg = LinearRegression()
reg.fit(x,y)


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(data,y,test_size=0.8)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[35]:


y_pred = reg.predict(x)
print("the value of predicted y is")
print(y_pred)


# In[36]:


rmse = np.sqrt((mean_squared_error(y_pred,y)))
print("the value of rmse is")
print(rmse)


# In[37]:


r2score = reg.score(x,y)
print("the value of r2score is")
print(r2score)


# In[ ]:




