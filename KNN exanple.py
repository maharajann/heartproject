#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics
from sklearn import datasets


# In[2]:


#Reading the csv file

data=pd.read_csv('heart.csv')


# In[3]:


x=data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y=data["target"]
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=30/100)
print(x_tr.shape)
print(x_ts.shape)
print(y_tr.shape)
print(y_ts.shape)


# In[4]:


KNN = knn(n_neighbors=5)
KNN.fit(x_tr, y_tr)
print(KNN)
y_pr = KNN.predict(x_ts)
print(y_pr)
print(y_ts)
print("Accuracy:",metrics.accuracy_score(y_ts, y_pr))


# In[5]:


print(KNN.predict([[56,0,1,140,294,0,0,153,0,1.3,1,0,2]]))


# In[6]:


print(KNN.predict([[55,1,0,132,353,0,1,132,1,1.2,1,1,3]]))


# In[ ]:




