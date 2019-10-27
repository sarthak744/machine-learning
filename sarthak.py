#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('hello world')


# In[2]:


print("hello")


# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('sar.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, :4].values
df.head()


# In[70]:


sns.heatmap(df.corr())


# In[71]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2 , random_state = 0 )


# In[72]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[73]:


y_pred = regressor.predict(x_test)
print(y_pred)



# In[74]:


print(regressor.coef_)


# In[75]:


print(regressor.intercept_)


# In[76]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

