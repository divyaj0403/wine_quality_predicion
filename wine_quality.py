#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[4]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


WineDF =pd.read_csv("winequality-red.csv")


# In[6]:


WineDF.head()


# In[7]:


WineDF.info()


# In[8]:


WineDF


# In[9]:


WineDF.describe()


# In[10]:


WineDF.columns


# In[11]:


sns.pairplot(WineDF)


# In[12]:


sns.heatmap(WineDF.corr(),annot=True)


# In[14]:


X=WineDF[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]

y=WineDF['quality']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=101)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()


# In[19]:


lm.fit (X_train, y_train)


# In[20]:


coeff_df = pd.DataFrame (lm.coef_, X.columns, columns=['Coefficient'])


# In[21]:


coeff_df


# In[22]:


predictions = lm.predict (X_test)


# In[23]:


plt.scatter (y_test, predictions)


# In[24]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:




