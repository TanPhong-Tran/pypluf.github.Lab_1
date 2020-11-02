

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn import preprocessing

# Import metrics to evaluate the perfomance of each model
from sklearn import metrics

# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import libraries nomalizer
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler


import os



df = pd.read_csv('AB_NYC_2019.csv')
df.head()


# # EDA

print(df.corr())


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot= True)
plt.yticks(rotation=0)

'''
Ta thấy không có sự tương quan giữa review_per_month và number_of_review
'''

# # Cleaning data


# Remove duplicate
df.drop_duplicates(subset = df.columns.values[:-1], keep = 'first', inplace = True)
print("Remove duplicate data: ", df.shape)
df.dropna()
print("Remove missing data: ", df.shape)


# In[7]:


# Remove outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df <(Q1 - 1.5 * IQR)) | (df > Q3 + 1.5 * IQR)).any(axis=1)]
print('Remove outlier: ', df.shape)

df_drop = df.drop(['name', 'host_name','neighbourhood_group', 'neighbourhood','last_review'], axis = 1)
df_drop.head()

df_drop.dtypes

y = df_drop['price']
X = df_drop.drop('price', axis = 1)
y.value_counts()



X.dtypes


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Cat_Colums = X.dtypes.pipe(lambda X: X[X=='float64']).index
for col in Cat_Colums:
    X[col] = label.fit_transform(X[col])
X.dtypes


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Cat_Colums = X.dtypes.pipe(lambda X: X[X=='object']).index
for col in Cat_Colums:
    X[col] = label.fit_transform(X[col])
X.dtypes


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)




l_reg = LinearRegression()
l_reg.fit(X_train,y_train)


predicts = l_reg.predict(X_test)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))



# # Scaling data


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


# # Build model after scaling

l_reg = LinearRegression()
l_reg.fit(X_train_normal,y_train)

predicts = l_reg.predict(X_test_normal)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))

l_reg = LinearRegression()
l_reg.fit(X_train_standard,y_train)

predicts = l_reg.predict(X_test_standard)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


l_reg = LinearRegression()
l_reg.fit(X_train_robust,y_train)

predicts = l_reg.predict(X_test_robust)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))


l_reg = LinearRegression()
l_reg.fit(X_train_minmax,y_train)

predicts = l_reg.predict(X_test_minmax)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test,predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
print("Mean Squareroot Error: ", mean_squared_error(y_test,predicts))

