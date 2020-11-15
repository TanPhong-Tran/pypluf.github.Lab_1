
# Import library
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None,'display.max_columns', None)
'''
# Plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')
'''


# Load dataset
path = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab1/Py4DS_Lab1_Dataset/mushrooms.csv'
df = pd.read_csv(path)
df.head()

# Import library for label encoder
from sklearn.preprocessing import LabelEncoder
# Transform data to numerical data
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.dtypes


# ## EDA


for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)


plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths=.1, cmap = "YlGnBu")
plt.yticks(rotation=0)
plt.savefig('Heatmap.png')




# ## Cleaning data

# Remove duplicate
df.drop_duplicates(subset = df.columns.values[:-1], keep = 'first', inplace = True)
print("Remove duplicate data: ", df.shape)
df.dropna()
print("Remove missing data: ", df.shape)


# Remove outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df <(Q1 - 1.5 * IQR)) | (df > Q3 + 1.5 * IQR)).any(axis=1)]
print('Remove outlier: ', df.shape)


# ## Split data


# Slitting dataset into training set and test set
# 70% training set and 30% test set
y = df['class']
X = df.drop('class', axis=1)
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, "\n\n", X_test.shape, "\n\n", y_train.shape, "\n\n", y_test.shape, "\n\n")


# ## Clustering

kms = KMeans(n_clusters= 7 , random_state= 0 ).fit(X_train)
y_pred = kms.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print("Accuracy using KMeans Clustering: ",metrics.accuracy_score(y_test, y_pred))

agg = AgglomerativeClustering(n_clusters = 2).fit(X_train)
y_pred = agg.fit_predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print("Accuracy using Agglomerative Clustering: ",metrics.accuracy_score(y_test, y_pred))

brc = Birch(n_clusters = 2).fit(X_train).fit(X_train)
y_pred = brc.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print("Accuracy using Birch Clustering: ",metrics.accuracy_score(y_test, y_pred))

'''
Accuracy using KMeans Clustering:  0.1875
Accuracy using Agglomerative Clustering:  0.7840909090909091
Accuracy using Birch Clustering:  0.7840909090909091
'''

