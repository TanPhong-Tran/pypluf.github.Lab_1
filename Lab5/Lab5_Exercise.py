'''
    Members:
        Tran Tan Phong - 18110181
        Vu Thien Nhan - 18110171
    Lab 5: Perform an Exploratory Data Analysis (EDA), Data cleaning, Building models for prediction,
    Presenting resultsusing on the following datasets

    Using Titanic passenger data to predict which passengers survived the Titanic disaster.
'''

# Import library
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Import ML models

from sklearn.svm import LinearSVC

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import scikit-learn metrics
from sklearn import metrics

# Function for Exploratory Data Analysis
def EDA(data):
   '''
        Purpose: Analy the data
        Input: data
        Output: The plot the show the correlation of data 
   '''
   # Botplot
   plt.figure(figsize=(15,7))
   sns.boxplot(data=data)
   plt.savefig('Boxplot_titanic.jpg')
   # Heatmap

   '''
        corr = data.corr()
        corr = corr.filter(items = [label])
        sns.heatmap(corr, annot = True)
   '''
   plt.figure(figsize = (14,12))
   sns.heatmap(data.corr(), linewidths=.1, cmap = "YlGnBu", annot = True)
   plt.yticks(rotation=0)
   plt.savefig('Heatmap_titanic.jpg')


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

# Label_Encoder function
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

# Function for LinearSVC models
def linearSVC(X_train, y_train, X_test, y_test):
   '''
        Purpose: Use linearSVC to calculate accuracy
        Input: X_train, y_train, X_test, y_test
        Output: accuracy_score
   '''
   clf = LinearSVC(random_state= 7)
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   return metrics.accuracy_score(y_test, y_pred)


def decision_tree(X_train,y_train,X_test,y_test):
   '''
        Purpose: Use linearSVC to calculate accuracy
        Input: X_train, y_train, X_test, y_test
        Output: accuracy_score
   '''
   clf = DecisionTreeClassifier(random_state= 7)
   clf = clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   return metrics.accuracy_score(y_test, y_pred)

# Main function
def main():
   ####### 1. LOAD DATASET #######
   
   # Load data
   path = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab5/titanic_train.csv'
   titanic_train = pd.read_csv(path)

   # Show the information of data
   print("\n\n####### 1. LOAD DATASET #######\n\n")
   print("Show the information of data")
   print("Print the 5 first line of data")
   print(titanic_train.head())
   print("Print the information of data")
   print(titanic_train.info())
   print("Caculate the sum of null value of data")
   print(titanic_train.isna().sum())
   print("Describe data")
   print(titanic_train.describe())
   
   
   ####### 2. HANDLE MISSING VALUE #######
   print("\n\n","~0~0"*27,"\n\n")
   print("####### 2. HANDLE MISSING VALUE #######")

   # Fill feature 'Age' with mean

   mean_age = round(titanic_train['Age'].mean())

   '''
   mean_cabin = round(titanic_train['Cabin'].mean())
   mean_embarked = round(titanic_train['Embarked'].mean())
   '''

   print("Filling with mean value of {}".format(mean_age))
   titanic_train['Age'] = titanic_train['Age'].fillna(mean_age)

   '''
   print("Filling with mean value of {}".format(mean_cabin))
   titanic_train['Cabin'] = sendy_data['Cabin'].fillna(mean_cabin)

   print("Filling with mean value of {}".format(mean_embarked))
   titanic_train['Embarked'] = sendy_data['Embarked'].fillna(mean_embaked)
   '''


   # Fill feature 'Cabin'
   print(titanic_train['Cabin'].value_counts())
   #titanic_train = titanic_train.drop('Cabin',axis=1)
   titanic_train['Cabin']=titanic_train['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else x )
   print(titanic_train['Cabin'].value_counts())
   titanic_train['Cabin']=titanic_train['Cabin'].apply(lambda x: x if pd.notnull(x) else 'C' )
   print(titanic_train['Cabin'].value_counts())

   # Fill feature 'Embarked'
   print(titanic_train['Embarked'].value_counts())
   titanic_train['Embarked']=titanic_train['Embarked'].fillna('S')

   print(titanic_train.info())
   
   titanic_train=label_encoder(titanic_train)

   ####### 3. EXPLORATORY DATA ANALYSIS ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 3. EXPLORATORY DATA ANALYSIS #######\n\n")
   print("Search plot in the same folder of this file\n")
   EDA(titanic_train)
   '''
   We see that the correlation between 'PassengerId' and 'Survived' is very low
   So we will drop the column 'PassengerId'
   '''
   print("After EDA, we drop PassengerId")
   titanic_train = titanic_train.drop('PassengerId',axis=1)
   
   
   ####### 4. DATA CLEANING ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 4. DATA CLEANING #######\n\n")

   ## Remove duplicate
   titanic_train.drop_duplicates(subset = titanic_train.columns.values[:-1], keep = 'first', inplace = True)
   print("Remove duplicate data: ",titanic_train.shape)

   titanic_train.dropna()
   print("Remove missing data: ", titanic_train.shape)


   ## Remove outlier
   titanic_train = remove_outlier(titanic_train)
   print('Remove outlier: ', titanic_train.shape)


   ####### 5. SPLIT DATA ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 5. SPLIT DATA #######\n\n")

   # We split data with test_size=0.1 and random_state=100
   print("We split data with test_size=0.1 and random_state=100")
   X = titanic_train.drop(['Survived'], axis=1)
   y = titanic_train['Survived']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)


   ####### 6. BUILD ML MODELS ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 6. BUILD ML MODELS #######\n\n")

   accuracy_SVC = linearSVC(X_train, y_train, X_test, y_test)
   print("Accuracy of Linear SVC:", accuracy_SVC)

   accuracy_decision=decision_tree(X_train,y_train,X_test,y_test)
   print("Accuracy of Decision Tree: ",accuracy_decision)

   '''
   Conclusion:
        Accuracy of Linear SVC: 0.7884615384615384
        Accuracy of Decision Tree:  0.75
   '''

if __name__ == '__main__':
    main()