

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


# Load dataset
path = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab1/Py4DS_Lab1_Dataset/xAPI-Edu-Data.csv'
df = pd.read_csv(path)
df.head()


# ## Int64


df.rename(index=str,columns={"gender":"Gender","NationalITy":"Nationality","raisedhands":"RaisedHands","VisITedResources":"VisitedResources"},inplace=True)
df.head()


for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)



sns.pairplot(df,hue = "StageID")
plt.show()
"""
Kiểm tra độ tương quan giữa các feagure thông qua các data point, ta thấy nếu phần trùng lập càng lớn thì độ tương quang giữa chúng càng cao
"""

P_Satis = sns.countplot(x = "Class", data = df, linewidth = 2, edgecolor = sns.color_palette("dark"))
plt.show()
"""
Ta thấy các lớp trong label Class có sự chênh lệch không quá con với M lớn nhất và L nhỏ nhất
""" 

sns.boxplot(x = 'Gender', y = 'RaisedHands', data = df)
plt.show()
"""
Từ boxplot ta thấy, số  lần giơ tay của Male và Female khá tương đồng với nhau 
"""

plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths=.1, cmap = "YlGnBu", annot = True)
plt.yticks(rotation=0)
plt.show()
"""
Các feagure có hệ số tương quan cao (>0.8), ta có thể kiểm tra sự tương quan giữa 2 figure để  thực hiện selection
"""



Facetgrid = sns.FacetGrid(df, hue = "Gender", size=6)
Facetgrid.map(sns.kdeplot,"RaisedHands", shade=True)
Facetgrid.set(xlim=(0, df['RaisedHands'].max()))
Facetgrid.add_legend()
plt.show()



sns.countplot(x = 'ParentschoolSatisfaction', hue = 'Class', data = df)
plt.show()



