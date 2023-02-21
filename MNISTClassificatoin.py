#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[5]:


data = datasets.load_digits()


# In[6]:


plt.subplot()
plt.imshow(data.images[0], cmap=plt.cm.gray_r)


# In[7]:


data.target[0]


# In[8]:


images_and_labels = list(zip(data.images, data.target))


# In[9]:


images_and_labels[0]


# In[10]:


for i, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, i+1)
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title(label)


# In[11]:


np.shape(data.images)


# In[12]:


len(data.images)


# In[13]:


X = data.images.reshape((len(data.images), -1))


# In[14]:


np.shape(X)


# In[15]:


X


# In[16]:


y = data.target


# In[17]:


len(y)


# In[18]:


from sklearn.svm import SVC
SVC


# In[47]:


svm_new = SVC()
svm_new.fit(X[:1000], y[:1000])


# In[48]:


svm_new.predict(X[1000:])


# In[49]:


y[1000:]


# In[50]:


from sklearn import metrics


# In[52]:


p = svm_new.predict(X[1000:])
e = y[1000:]
print(metrics.classification_report(e, p))


# In[53]:


print(metrics.confusion_matrix(e, p))

