#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


data = datasets.load_digits()


# In[3]:


plt.subplot()
plt.imshow(data.images[0], cmap=plt.cm.gray_r)


# In[4]:


data.target[0]


# In[5]:


images_and_labels = list(zip(data.images, data.target))


# In[6]:


images_and_labels[0]


# In[34]:


for i, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, i+1)
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title(label)


# In[8]:


np.shape(data.images)


# In[9]:


len(data.images)


# In[10]:


X = data.images.reshape((len(data.images), -1))


# In[11]:


np.shape(X)


# In[12]:


X


# In[13]:


y = data.target


# In[14]:


len(y)


# In[15]:


from sklearn.svm import SVC
SVC


# In[27]:


np.unique(y)


# In[28]:


svm_new = SVC()
svm_new.fit(X[:1000], y[:1000])


# In[29]:


svm_new.predict(X[1000:])


# In[30]:


y[1000:]


# In[31]:


from sklearn import metrics


# In[32]:


p = svm_new.predict(X[1000:])
e = y[1000:]
print(metrics.classification_report(e, p))


# In[33]:


print(metrics.confusion_matrix(e, p))

