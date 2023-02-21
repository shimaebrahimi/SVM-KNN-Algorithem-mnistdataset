#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


data=datasets.load_digits()


# In[3]:


data


# In[4]:


x=data.images.reshape(len(data.images),-1)
x
y=data.target


# In[5]:


x


# In[6]:


y


# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[10]:


knn_cll=KNeighborsClassifier(n_neighbors=3)


# In[12]:


knn_cll.fit(x[:1000],y[:1000])


# In[13]:


p=knn_cll.predict(x[1000:])
e=data.target[1000:]


# In[14]:


from sklearn import metrics


# In[15]:


print(metrics.classification_report(e,p))


# In[16]:


for i in range(1,101,1):
    knn_cll=KNeighborsClassifier(n_neighbors=i)
    knn_cll.fit(x[:1000],y[:1000])
    p=knn_cll.predict(x[1000:])
    e=data.target[1000:]
    z=metrics.accuracy_score(e,p)
    print(i,z)


# In[ ]:




