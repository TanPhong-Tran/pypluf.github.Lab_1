'''
    Members:
        Tran Tan Phong - 18110181
        Vu Thien Nhan - 18110171
    Exercise 3:
    1. Defining the functions correspond with the feature selection or dimensionality reduction
technologies.
    2. Giving examples to demonstrate your function works
'''

## IMPORT LIBRARY

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# import and create the VarianceThreshold object.
from sklearn.feature_selection import VarianceThreshold

# import the required functions and object.
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

# import the required functions and object.
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
## FUNCTION TO REMOV CONSTANT FEATURE USING VARIANCE THRESHOLD

# Remove outlier function
def remove_outlier(data):
   '''
        Purpose: Remove outlier
        Input: data
        Output: the encoded DataFrame 
   '''
   Q1 = data.quantile(0.25)
   Q3 = data.quantile(0.75)
   IQR = Q3 - Q1
   data_after = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
   print('Shape of data before handling outlier values: ', data.shape)
   print('Shape of data after handling outlier values: ', data_after.shape)
   return data_after   
def label_encoder(data):
    '''
        Purpose: Encode label for data, convert to numeric type
        Input: data
        Output: the encoded DataFrame 
    '''
    label = LabelEncoder()
    data_colums = data.dtypes.pipe(lambda X: X[X=='object']).index
    for col in data_colums:
        data[col] = label.fit_transform(data[col])
    return data
def Constant_Features(X_train, X_test,threshold = 0):
  """
  Removing Constant Features using Variance Threshold
  Input: threshold parameter to identify the variable as constant
         train data (pd.Dataframe) 
         test data (pd.Dataframe)
  Output: train data, test data after applying filter methods
  """
  
  vs_constant = VarianceThreshold(threshold)

  # select the numerical columns only.
  numerical_X_train = X_train[X_train.select_dtypes([np.number]).columns]

  # fit the object to our data.
  vs_constant.fit(numerical_X_train)

  # get the constant colum names.
  constant_columns = [column for column in numerical_X_train.columns
                      if column not in numerical_X_train.columns[vs_constant.get_support()]]

  # detect constant categorical variables.
  constant_cat_columns = [column for column in X_train.columns 
                          if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]

  # concatenating the two lists.
  all_constant_columns = constant_cat_columns + constant_columns

  # drop the constant columns
  X_train = X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
  X_test = X_test.drop(labels=all_constant_columns, axis=1, inplace=True)
  return X_train, X_test

## 
def Quasi_Constant_Features(X_train, X_test, threshold=0.98):
    """
    Show feature selection using Quasi-Constant
    Input: threshold parameter to identify the variable as constant
         train data (pd.Dataframe) 
         test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # create empty list
    quasi_constant_feature = []

    # loop over all the columns
    for feature in X_train.columns:

        # calculate the ratio.
        predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
    
        # append the column name if it is bigger than the threshold
        if predominant >= threshold:
            quasi_constant_feature.append(feature)   
        
    print(quasi_constant_feature)

    # drop the quasi constant columns
    X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
    X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True) 
    return X_train, X_test


def Duplicate_Features(X_train, X_test):
    # transpose the feature matrice
    train_features_T = X_train.T

    # print the number of duplicated features
    print(train_features_T.duplicated().sum())

    # select the duplicated features columns names
    duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

    # drop those columns
    X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
    X_test.drop(labels=duplicated_columns, axis=1, inplace=True)
    return X_train, X_test
    

def Correlation_Filter_Methods(X_train, X_test):
    # creating set to hold the correlated features
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = X_train.corr()

    # optional: display a heatmap of the correlation matrix
    plt.figure(figsize=(11,11))
    sns.heatmap(corr_matrix)

    for i in range(len(corr_matrix .columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
                
    X_train.drop(labels=corr_features, axis=1, inplace=True)
    X_test.drop(labels=corr_features, axis=1, inplace=True)
    return X_train, X_test


def Mutual_Information(X_train, y_train, X_test):
    # select the number of features you want to retain.
    select_k = 10

    # get only the numerical features.
    numerical_X_train = X_train[X_train.select_dtypes([np.number]).columns]


    # create the SelectKBest with the mutual info strategy.
    selection = SelectKBest(mutual_info_classif, k=select_k).fit(numerical_X_train, y_train)

    # display the retained features.
    features = X_train.columns[selection.get_support()]
    #print(features)
    return X_train[features], X_test[features]

def Chi_squared_Score(X_train, y_train, X_test):
    # change this to how much features you want to keep from the top ones.
    select_k = 10

    # apply the chi2 score on the data and target (target should be binary).  
    selection = SelectKBest(chi2, k=select_k).fit(X_train, y_train)

    # display the k selected features.
    features = X_train.columns[selection.get_support()]
    #print(features)
    return X_train[features], X_test[features]

def PCA_method(X_train, X_test):
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

def RFE_method(X_train, X_test, y_train, y_test):
    #def RFE_method(X_train, y_train):
    # define model
    rfc = RandomForestClassifier(n_estimators=100)
    rfe = RFE(estimator=rfc, n_features_to_select=2)
    # fit the model
    rfe.fit(X_train, y_train)
    # transform the data
    #X_train, y_train = rfe.transform(X_train, y_train)
    #X_test, y_test = rfe.transform(X_test, y_test)
    X_train= rfe.transform(X_train)
    X_test = rfe.transform(X_test)
    #return X_train, y_train, X_test, y_test
    return X_train, X_test


def main():
    ####### 0.1. LOAD DATASET #######
    df = pd.read_csv('/home/nhan/Downloads/data.csv')    
    print("\n\n >>>>>>> dataframe:\n\n")
    print(df)
    print("\n\nThe info of data: \n\n")
    print(df.info())
    print("\n\n >>>>>>> Describe the data: \n\n")
    print(df.describe())

    ####### 0.2. DATA CLEANING ####### 
    print("\n\n","~0~0"*27,"\n\n")
    print("\n\n####### 0.2. DATA CLEANING #######\n\n")
     ## Remove duplicate
    df.drop_duplicates(subset = df.columns.values[:-1], keep = 'first', inplace = True)
    print("Remove duplicate data: ",df.shape)

    df.dropna()
    print("Remove missing data: ", df.shape)

    ## Remove outlier
    df = remove_outlier(df)
    print('Remove outlier: ', df.shape)
    
    ####### 0.3 SPLIT DATA #######
    print("\n\n####### 0.3 SPLIT DATA   #######\n\n")
    y = df['International Reputation']
    X = df.drop('International Reputation', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    print("\n\nShow X_train, X_test, y_train, y_test\n\n")
    print("\n\n>>>>>>> X_train\n\n")
    print(X_train)
    print("\n\n>>>>>>> y_train\n\n")
    print(y_train)
    print("\n\n>>>>>>> X_test\n\n")
    print(X_test)
    print("\n\n>>>>>>> y_test\n\n")
    print(X_test)

    ####### 1. APPLY CONSTANT FEATURES #######
    print("\n\n####### 1. APPLY CONSTANT FEATURES  #######\n\n")
    X_train_confea, X_test_confea = Quasi_Constant_Features(X_train, X_test, threshold=0.98)
    print("\n\nShow X_train, X_test after apply Quasi-constant features filter\n\n")
    print("\n\n>>>>>>> X_train after apply Quasi-constant features filter\n\n")
    print(X_train_confea)
    print("\n\n>>>>>>> X_test after apply Quasi-constant features filter\n\n")
    print(X_test_confea)

    ####### 2. APPLY CORRELATION FILTER METHOD #######
    print("\n\n####### 2. APPLY CORRELATION FILTER METHOD  #######\n\n")
    X_train_cor, X_test_cor = Correlation_Filter_Methods(X_train, X_test)
    print("\n\nShow X_train, X_test after apply Quasi-constant features filter\n\n")
    print("\n\n>>>>>>> X_train after apply Corralation filter method\n\n")
    print(X_train_cor)
    print("\n\n>>>>>>> X_test after apply Corralation filter methodr\n\n")
    print(X_test_cor)



if __name__ == '__main__':
	main()