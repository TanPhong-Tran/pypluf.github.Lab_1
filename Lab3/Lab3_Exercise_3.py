
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


df = pd.read_csv('FIFA2018Statistics.csv')
df.head()


df.dtypes.head()



df.isnull().sum().head()


df.dtypes.head()


# # Exploratory data analysis (EDA)

for i in range(1,27):
    print(df.iloc[:,i].value_counts())
    print("*"*20)


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu")
plt.yticks(rotation=0)

'''
'Man of the match' có sự tương quan với 'Goal Scored', 'On-Target','Man of the Match' is highly correlated with 'Goal Scored', 'On-Target', 'Corners', 'Attempts', 'free Kicks', 'Yellow Card', 'red', 'Fouls Committed', 'Own goal Time'
'''

P_Satis = sns.countplot(x="Man of the Match",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))
'''
Man of the match có dữ liệu cân bằng
'''

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


# # Change to int 64


df_drop = df.drop(['1st Goal', 'Own goals','Own goal Time','On-Target','Corners','Fouls Committed'], axis = 1)
df_drop.head()


df_drop.dtypes



y = df_drop['Man of the Match']
X = df_drop.drop('Man of the Match',axis = 1)
y.value_counts()



from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Cat_Colums = X.dtypes.pipe(lambda X: X[X=='object']).index
for col in Cat_Colums:
    X[col] = label.fit_transform(X[col])
X.dtypes


# 

# # Build Model


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)
# Predict testset
y_pred=rdf.predict(X_test)
# Evaluate performance of the model
print("RDF:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")


# # Scalling data



def scaling_data(X_train, X_test, method='None'):
    if method == 'Normalizer':
        X_scale = Normalizer() 
    elif method == 'StandardScaler':
        X_scale = StandardScaler()
    elif method == 'MinMaxScaler':
        X_scale = MinMaxScaler()
    elif method == 'RobustScaler':
        X_scale = RobustScaler()
    X_scale.fit(X_train)
    X_train_scaled = X_scale.transform(X_train)
    X_scale.fit(X_test)
    X_test_scaled = X_scale.transform(X_test)
    return X_train_scaled, X_test_scaled


X_train_normal, X_test_normal = scaling_data(X_train, X_test, method='Normalizer')

X_train_standard, X_test_standard = scaling_data(X_train, X_test, method='StandardScaler')

X_train_robust, X_test_robust = scaling_data(X_train, X_test, method='RobustScaler')

X_train_minmax, X_test_minmax = scaling_data(X_train, X_test, method='MinMaxScaler')


# # Buiding model after scaling

# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_normal, y_train)
# Predict testset
y_pred=rdf.predict(X_test_normal)
# Evaluate performance of the model
print("RDF:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")



# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_standard, y_train)
# Predict testset
y_pred=rdf.predict(X_test_standard)
# Evaluate performance of the model
print("RDF:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")



# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_robust, y_train)
# Predict testset
y_pred=rdf.predict(X_test_robust)
# Evaluate performance of the model
print("RDF:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")


# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train_minmax, y_train)
# Predict testset
y_pred=rdf.predict(X_test_minmax)
# Evaluate performance of the model
print("RDF:  ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")

