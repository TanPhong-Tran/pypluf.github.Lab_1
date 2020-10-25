#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


# In[3]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[5]:


for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)


# In[4]:


V = df[[col for col in df.columns if 'V' in col]]

f, ax = plt.subplots(ncols = 2, nrows = 14, figsize=(15,2*len(V.columns)))


for i, c in zip(ax.flatten(), V.columns):
    sns.distplot(V[c], ax = i)

f.tight_layout()


# In[11]:


P_Satis = sns.countplot(x = "Class", data = df, linewidth = 2, edgecolor = sns.color_palette("dark"))
plt.show()
"""
Nhìn vào đồ thị, ta thấy dữ liệu không cân bằng và có sự chênh lệch lớn
"""


# In[12]:


df.Class.value_counts(normalize = True).plot(kind = "bar")
plt.show()


# In[29]:


plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths=.1, cmap = "YlGnBu", annot = True)
plt.yticks(rotation=0)
plt.show()


# In[14]:


df.Class.value_counts()


# In[15]:


df.Class.value_counts(normalize = True)


# In[18]:


Facegrid = sns.FacetGrid(df, hue='Class', size=6) 
Facegrid.map(sns.kdeplot, 'Time', shade=True) #
Facegrid.add_legend()
plt.show()
"""
Từ đồ thị ta thấy có sự sư biến đổi đột biến trong 2 khoảng (20000,100000), (120000,175000)
"""


# In[5]:


Amount = sns.boxplot(x="Class", y="Amount", data=df)
Amount.set(ylim=(df['Amount'].min(),300))
plt.show()
"""
Boxplot cho ta thấy đồ thi bị lệch, không có sự cân bằng.
"""


# In[ ]:




