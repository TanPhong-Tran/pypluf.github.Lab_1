


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



for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)

    
sns.pairplot(df,hue = "Class")
plt.show()
"""
Kiểm tra độ tương quan giữa các feagure thông qua các data point, ta thấy nếu phần trùng lập càng lớn thì độ tương quang giữa chúng càng cao
"""


P_Satis = sns.countplot(x = "Class", data = df, linewidth = 2, edgecolor = sns.color_palette("dark"))
plt.show()
"""
Ta thấy các lớp trong label Class có sự chênh lệch không quá con với M lớn nhất và L nhỏ nhất
"""

plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths=.1, cmap = "YlGnBu", annot = True)
plt.yticks(rotation=0)
plt.show()
"""
Các feagure có hệ số tương quan cao (>0.8), ta có thể kiểm tra sự tương quan giữa 2 figure để  thực hiện selection
"""



df.Class.value_counts()
df.Class.value_counts(normalize = True)
"""
Ta so sánh dữ trên tỉ lệ phần trăm
"""



plt.subplots(figsize=(20,8))
df["raisedhands"].value_counts().sort_index().plot.bar()
plt.title("No. of times", fontsize=18)
plt.xlabel("No. of times, student raised their hand", fontsize=14)
plt.ylabel("No. of student, on particular times", fontsize=14)
plt.show()



Raised_hand = sns.boxplot(x = "Class", y = "raisedhands", data=df)
plt.show()
"""
Lớp H (60,90), Lớp M(25,75),  Lớp L: <30
=> Vậy số  học sinh giỏi giơ tay phát biểu nhiều, số học sinh kém giơ tay phát biểu ít 
"""


Facetgrid = sns.FacetGrid(df, hue = "Class", size=6)
Facetgrid.map(sns.kdeplot,"raisedhands", shade=True)
Facetgrid.set(xlim=(0, df['raisedhands'].max()))
Facetgrid.add_legend()
plt.show()



labels = df.ParentschoolSatisfaction.value_counts()
colors=["blue","green"]
explode = [0,0]
sizes = df.ParentschoolSatisfaction.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct='%1.1f%%')
plt.title("Parents school Satisfaction in Data", fontsize = 15)
plt.show()




