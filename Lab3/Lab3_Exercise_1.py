


# Import libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt


# Import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import metrics to evaluate the perfomance of each model
from sklearn import metrics

# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import libraries nomalizer
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler

import os





df = pd.read_csv('creditcard.csv')
df.head()



df.describe()


df.dtypes


# # EDA

print(df.corr())
plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlOrRd")
plt.yticks(rotation=0)

'''
Không có sự tương quan rõ rệt giữa các features
'''

P_Satis = sns.countplot(x="Class",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))
'''
Dữ liệu bị mất cân bằng nên ta cân bằng dữ liệu
'''

# # Balance Data

fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(len(fraud) * 5)
non_fraud.reset_index(drop=True, inplace=True)
fraud.reset_index(drop=True, inplace=True)
df= pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)
df.describe()


# # Build Model

y = df['Class']
X = df.drop('Class',axis = 1)
y.value_counts()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# clf = DecisionTreeClassifier(criterion='entropy')
clf = DecisionTreeClassifier()

# Fit Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict testset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))

# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")


# # Cleaning data


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


# # Scaling data


def scaling_data(x_train, x_test, method='None'):
    if method == 'Normalizer':
        x_scale = Normalizer() 
    elif method == 'StandardScaler':
        x_scale = StandardScaler()
    elif method == 'MinMaxScaler':
        x_scale = MinMaxScaler()
    elif method == 'RobustScaler':
        x_scale = RobustScaler()
    x_scale.fit(x_train)
    x_train_scaled = x_scale.transform(x_train)
    x_scale.fit(x_test)
    x_test_scaled = x_scale.transform(x_test)
    return x_train_scaled, x_test_scaled


X_train_normal, X_test_normal = scaling_data(X_train, X_test, method='Normalizer')

X_train_standard, X_test_standard = scaling_data(X_train, X_test, method='StandardScaler')

X_train_robust, X_test_robust = scaling_data(X_train, X_test, method='RobustScaler')

X_train_minmax, X_test_minmax = scaling_data(X_train, X_test, method='MinMaxScaler')


# # Build model after scaling
# 

# Fit Decision Tree Classifier
clf = clf.fit(X_train_normal, y_train)
# Predict testset
y_pred = clf.predict(X_test_normal)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))

# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")


# Fit Decision Tree Classifier
clf = clf.fit(X_train_standard, y_train)
# Predict testset
y_pred = clf.predict(X_test_standard)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))

# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")

# Fit Decision Tree Classifier
clf = clf.fit(X_train_robust, y_train)
# Predict testset
y_pred = clf.predict(X_test_robust)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))

# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")



# Fit Decision Tree Classifier
clf = clf.fit(X_train_minmax, y_train)
# Predict testset
y_pred = clf.predict(X_test_minmax)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy:  {}".format(sum(y_pred == y_test) / len(y_pred)))  # equivalent to the next row
print("CART (Tree Prediction) Accuracy by calling metrics:  ", metrics.accuracy_score(y_test, y_pred))

# Evaluate a score by cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")



