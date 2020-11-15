# Import library
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MeanShift
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
#Path of dataset
path = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab1/Py4DS_Lab1_Dataset/diabetes.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
dataset_pd = pd.read_csv(path,header = 0, names = col_names)
dataset_np=np.genfromtxt(path,delimiter=',')


dataset_pd



dataset_pd.head()



print(dataset_pd.head(5))



#Check basic metadata
print(dataset_pd.shape)
print(dataset_np.shape)


#EDA
'''
for i in range(1,17):
    print(dataset_pd.iloc[:,i].value_counts())
    print("*"*20)
'''
plt.figure(figsize=(14,12))
sns.heatmap(dataset_pd.corr(),linewidths=.1,cmap="YlGnBu")
plt.yticks(rotation=0)
plt.savefig('Heatmap.png')


# Remove duplicate
dataset_pd.drop_duplicates(subset = dataset_pd.columns.values[:-1], keep = 'first', inplace = True)
print("Remove duplicate data: ", dataset_pd.shape)
dataset_pd.dropna()
print("Remove missing data: ", dataset_pd.shape)


# Remove outlier
Q1 = dataset_pd.quantile(0.25)
Q3 = dataset_pd.quantile(0.75)
IQR = Q3 - Q1
dataset_pd = dataset_pd[~((dataset_pd <(Q1 - 1.5 * IQR)) | (dataset_pd > Q3 + 1.5 * IQR)).any(axis=1)]
print('Remove outlier: ', dataset_pd.shape)


# Separate between feature (X) and label (y)
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = dataset_pd[feature_cols]
y = dataset_pd.label



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, "\n\n", X_test.shape, "\n\n", y_train.shape, "\n\n", y_test.shape, "\n\n")


kms = KMeans(n_clusters= 7 , random_state= 0 ).fit(X_train)
y_pred = kms.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print("Accuracy using KMeans Clustering: ",metrics.accuracy_score(y_test, y_pred))

agg = AgglomerativeClustering(n_clusters = 1).fit(X_train)
y_pred = agg.fit_predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print("Accuracy using Agglomerative Clustering: ",metrics.accuracy_score(y_test, y_pred))

brc = Birch(n_clusters = 2).fit(X_train).fit(X_train)
y_pred = brc.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print("Accuracy using Birch Clustering: ",metrics.accuracy_score(y_test, y_pred))

'''
Accuracy using KMeans Clustering:  0.17061611374407584
Accuracy using Agglomerative Clustering:  0.6777251184834123
Accuracy using Birch Clustering:  0.5450236966824644
'''


