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
#def linearSVC(X_train, y_train, X_test, y_test):
def linearSVC(X_train, y_train, X_test):
   '''
        Purpose: Use linearSVC to predict
        Input: X_train, y_train, X_test
        Output: y_pred
   '''
   clf = LinearSVC(random_state= 777)
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   #return metrics.accuracy_score(y_test, y_pred)
   return y_pred


#def decision_tree(X_train,y_train,X_test,y_test):
def decision_tree(X_train,y_train,X_test):
   '''
        Purpose: Use linearSVC to predict
        Input: X_train, y_train, X_test, y_test
        Output: y_pred
   '''
   clf = DecisionTreeClassifier(random_state= 777)
   clf = clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   #return metrics.accuracy_score(y_test, y_pred)
   return y_pred

def main():
   ####### 1. LOAD DATASET #######
   
   # Load data
   path = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab5/titanic_train.csv'
   titanic_train = pd.read_csv(path)
   path_1 = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab5/titanic_test.csv'
   titanic_test = pd.read_csv(path_1)
   '''
   path_2 = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Lab5/gender_submission.csv'
   gender = pd.read_csv(path_2)
   '''

   # Show the information of data
   print("\n\n####### 1. LOAD DATASET #######\n\n")
   print("\n\nShow the information of data\n\n")
   print("\n\nPrint the 5 first line of data\n\n")
   print(titanic_test.head())
   print("\n\nPrint the information of data\n\n")
   print(titanic_test.info())
   print("\n\nCaculate the sum of null value of data\n\n")
   print(titanic_test.isna().sum())
   print("\n\nDescribe data\n\n")
   print(titanic_test.describe())
   
   
   ####### 2. HANDLE MISSING VALUE #######
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 2. HANDLE MISSING VALUE #######\n\n")

   # Fill feature 'Age' with mean
   mean_age = round(titanic_train['Age'].mean())
   mean_age = round(titanic_test['Age'].mean())

   '''
   mean_cabin = round(titanic_test['Cabin'].mean())
   mean_embarked = round(titanic_test['Embarked'].mean())
   '''

   print("Filling age of titanic_train with mean value of {}".format(mean_age))
   titanic_train['Age'] = titanic_train['Age'].fillna(mean_age)

   print("Filling age of titanic_test with mean value of {}".format(mean_age))
   titanic_test['Age'] = titanic_test['Age'].fillna(mean_age)

   '''
   print("Filling with mean value of {}".format(mean_cabin))
   titanic_test['Cabin'] = sendy_data['Cabin'].fillna(mean_cabin)

   print("Filling with mean value of {}".format(mean_embarked))
   titanic_test['Embarked'] = sendy_data['Embarked'].fillna(mean_embaked)
   '''


   # Fill feature 'Cabin'
   print(titanic_train['Cabin'].value_counts())
   #titanic_train = titanic_train.drop('Cabin',axis=1)
   titanic_train['Cabin']=titanic_train['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else x )
   print(titanic_train['Cabin'].value_counts())
   titanic_train['Cabin']=titanic_train['Cabin'].apply(lambda x: x if pd.notnull(x) else 'C' )
   print(titanic_train['Cabin'].value_counts())


   print(titanic_test['Cabin'].value_counts())
   #titanic_test = titanic_test.drop('Cabin',axis=1)
   titanic_test['Cabin']=titanic_test['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else x )
   print(titanic_test['Cabin'].value_counts())
   titanic_test['Cabin']=titanic_test['Cabin'].apply(lambda x: x if pd.notnull(x) else 'C' )
   print(titanic_test['Cabin'].value_counts())

   # Fill feature 'Embarked'
   print(titanic_train['Embarked'].value_counts())
   titanic_train['Embarked']=titanic_train['Embarked'].fillna('S')


   print(titanic_test['Embarked'].value_counts())
   titanic_test['Embarked']=titanic_test['Embarked'].fillna('S')

   print(titanic_train.info())
   print(titanic_test.info())
   
   titanic_train=label_encoder(titanic_train)
   titanic_test=label_encoder(titanic_test)

   # Fill feature 'Fare'
   mean_fare = round(titanic_test['Fare'].mean())

   print("Filling Fare of titanic_test with mean value of {}".format(mean_age))
   titanic_test['Fare'] = titanic_test['Fare'].fillna(mean_fare)

   ####### 3. EXPLORATORY DATA ANALYSIS ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 3. EXPLORATORY DATA ANALYSIS #######\n\n")
   print("Search plot in the same folder of this file\n")
   EDA(titanic_test)
   titanic_train = titanic_train.drop('PassengerId',axis=1)
   titanic_test = titanic_test.drop('PassengerId',axis=1)
   '''
   We see that the correlation between 'PassengerId' and 'Survived' is very low
   So we will drop the column 'PassengerId'
   '''
   
   ####### 4. DATA CLEANING ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 4. DATA CLEANING #######\n\n")

   ## Remove duplicate
   titanic_test.drop_duplicates(subset = titanic_test.columns.values[:-1], keep = 'first', inplace = True)
   print("Remove duplicate data: ",titanic_test.shape)

   titanic_test.dropna()
   print("Remove missing data: ", titanic_test.shape)

   
   ## Remove outlier
   titanic_train = remove_outlier(titanic_train)
   print('Remove outlier: ', titanic_train.shape)

   print("titanic_train is used to build model\n")
   print(titanic_train.info())
   print("titanic_test is used to build model\n")
   print(titanic_test.info())
   

   ####### 5. SPLIT DATA ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 5. SPLIT DATA #######\n\n")


   X_train = titanic_train.drop(['Survived'], axis=1)
   y_train = titanic_train['Survived']
   X_test = titanic_test
   #y_test = gender.drop(['PassengerId'], axis=1)
   #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)


   ####### 6. BUILD ML MODELS ####### 
   print("\n\n","~0~0"*27,"\n\n")
   print("\n\n####### 6. BUILD ML MODELS #######\n\n")

   #accuracy_SVC = linearSVC(X_train, y_train, X_test, y_test)    
   y_predict_SVC = linearSVC(X_train, y_train, X_test)
   #print("Accuracy of Linear SVC:", accuracy_SVC)
   print("y-predict of Linear SVC:", y_predict_SVC)

   #accuracy_decision=decision_tree(X_train,y_train,X_test,y_test)
   y_predict_decision=decision_tree(X_train,y_train,X_test)
   #print("Accuracy of Decision Tree: ",accuracy_decision)
   print("y-predict of Decision Tree: ",y_predict_decision)

   '''
   Conclusion:
        y-predict of Linear SVC: 
        
        [0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 1 0 1 0 0 0 0 0 0 0 1 0 0
 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 0 0 0
 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0
 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 1 1 1 1 0 1 1 0 1
 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 1 0 1 1 0 0 0
 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0
 1 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 0 1 1 0 0 1 1 0 1 1 0
 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 0 0
 1 0 0 0 1 0 0 1 0 0 0]
        y-predict of Decision Tree:  
        
        [0 0 0 0 1 1 0 1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 0 0 1 0
 1 0 0 1 0 0 1 0 0 0 0 1 1 0 1 1 1 0 0 0 1 0 1 0 0 1 0 0 1 0 0 1 0 1 0 0 1
 1 0 0 1 0 0 0 0 1 0 0 0 0 1 1 0 1 1 1 0 1 0 1 0 1 0 0 1 0 1 1 1 1 0 0 0 0
 1 1 0 0 0 0 1 0 1 1 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0
 0 1 1 0 0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 1 0 1
 0 1 0 0 0 1 1 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0
 1 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1
 1 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 1 0 0 0 1 0 1 0 0 0 0 0 0
 1 1 0 0 1 0 0 0 1 1 0 1 0 1 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0
 1 0 1 0 0 0 0 0 1 0 1 1 1 0 0 0 1 0 0 0 1 0 1 1 0 0 1 0 1 1 0 1 0 0 0 0 1
 0 1 0 0 1 1 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0
 0 1 1 1 1 0 1 0 0 0 0]
   '''

if __name__ == '__main__':
    main()