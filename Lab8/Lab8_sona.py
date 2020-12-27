'''
    Lab 8: Using model selection technical to discover a well-performing model configuration for the sonar dataset
dataset.
'''

#Import necessary library
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#Principal Component Analysis Function
def PCA_method(data):
    '''
        Purpose: Perform PCA
        Input: Dataframe
        Output: X_pca
    '''
    pca = PCA(n_components=2)
    pca.fit(data)
    X_pca = pca.transform(data)
    return X_pca

#Label Encoder Function
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

# Main function
def main():

    #Load data
    url_data='https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
    data=pd.read_csv(url_data,header=None)
    print("\n\n####### 1. LOAD DATASET #######\n\n")
    print("\n\nPrint the 5 first line of data\n\n")
    print(data.head())
    print("\n\nthe information of data\n\n")
    print(data.info())
    print("\n\n Describe data\n\n")
    print(data.describe())
    print('\n\nShape of the dataset: \n\n')
    print(data.shape)
    print('='*80)


    ####### 2. LABEL ENCODER ####### 
    print("\n\n####### 2. LABEL ENCODER #######\n\n")
    
    #Use label encoder function
    data=label_encoder(data)
    print("\n\nthe information of data after use Label Encoder function\n\n")
    print(data.info())

    print('='*80)

    ####### 3. CLEAN DATA ####### 
    print("\n\n","~0~0"*27,"\n\n")
    print("\n\n####### 2. CLEAN DATA #######\n\n")
    #Perform outlier-removing
    data=remove_outlier(data)
    print('='*80)

    ####### 4. SPLIT DATA ####### 
    print("\n\n####### 4. SPLIT DATA  #######\n\n")
    #Split Data
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    print("\n\nPrint the 5 first line of X\n\n")
    print(X.head())
    print("\n\nPrint the 5 first line of y\n\n")
    print(y.head())
    print('='*80)


    ####### 5. USE PRINCIPAL COMPONENT ANALYSIS (PCA) ####### 
    print("\n\n####### 5. USE PRINCIPAL COMPONENT ANALYSIS (PCA) #######\n\n")    
    #Do Principal Component Analysis as pca
    print('Perform Principal Component Analysis')
    X_pca=PCA_method(X)
    print('\n\nShape of dataset before PCA: \n\n')
    print(X.shape)
    print('\n\nShape of dataset after PCA: \n\n')
    print(X_pca.shape)
    print('='*100)
    print('='*80)

    ####### 6. PERFORM GRIBSEARCHCV ####### 
    print("\n\n####### 6. PERFORM GRIBSEARCHCV #######\n\n") 

    #Define model
    model = Ridge()

    #Define evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    #Define search space
    space = dict()
    space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
    space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    space['fit_intercept'] = [True, False]
    space['normalize'] = [True, False]

    #Define search
    search = GridSearchCV(model, space, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
    result = search.fit(X_pca, y)

    #Result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    print('='*80)
    '''
        Summary:
            Best Score: -0.1618722594173935
            Best Hyperparameters: {'alpha': 0.1, 'fit_intercept': True, 'normalize': True, 'solver': 'lsqr'}
    '''

if __name__=='__main__':
    main()