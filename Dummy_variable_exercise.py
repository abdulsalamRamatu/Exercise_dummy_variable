#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("C:\\Users\\Ramatu\\.jupyter\\py-master\\ML\\5_one_hot_encoding\\Exercise\\carprices.csv")
df.head()


# In[3]:


dummies=pd.get_dummies(df.CarModel)
dummies.head()


# In[5]:


merged=pd.concat([df,dummies], axis='columns')
merged.head()


# In[6]:


done=merged.drop(['CarModel', 'Mercedez Benz C class'], axis='columns')
done.head()


# In[7]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[13]:


X=done.drop('SellPrice', axis='columns')
X.head()


# In[14]:


y=done.SellPrice
y.head()


# In[15]:


model.fit(X,y)


# In[18]:


model.predict([[45000,4,0,0]])


# In[19]:


model.predict([[86000,7,0,1]])


# In[20]:


model.score(X,y)


# In[ ]:




