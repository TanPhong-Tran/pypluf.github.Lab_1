
# # Ex3_Mushroom_Classification 
#

# Import libraries 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



#import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier



# Import train_test_split function
from sklearn.model_selection import train_test_split


# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Load dataset
path = 'https://raw.githubusercontent.com/pypluf/pypluf.github.Lab_1/master/Py4DS_Lab1_Dataset/mushrooms.csv'
df = pd.read_csv(path)
df.head()


# Slitting dataset into training set and test set
# 70% training set and 30% test set
y = df['class']
X = df.drop('class', axis=1)
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Import library for label encoder
from sklearn.preprocessing import LabelEncoder
# Transform data to numerical data
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.dtypes

df.head()

df.describe()


## Decision Tree Classifier

#clf = DecisionTreeClassifier(criterion='emtropy')
clf = DecisionTreeClassifier()
# Fit Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy: {}".format(sum(y_pred == y_test) / len(y_pred)))
print("CART (Tree Prediction) Accuracy by calling metrics: ", metrics.accuracy_score(y_test, y_pred))

# Evaluate a score by cross validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))


## Random Forest Classifier
 
#Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf = rdf.fit(X_train, y_train)
#Predict testset
y_pred = rdf.predict(X_test)
#Evaluate performance of the model
print("Accuracy of RDF: ", metrics.accuracy_score(y_test, y_pred))
print("\n")
# Evaluate a score by cross validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")

## Logistic Regression Classifier

#Fit Logistic Regression Classifier
lr = LogisticRegression(max_iter=2000)
lr = lr.fit(X_train, y_train)
#Predict testset
y_pred = lr.predict(X_test)
#Evaluate performance of the model
print("Accuracy of LR: ",metrics.accuracy_score(y_test,y_pred))
print("\n")
#Evaluate a score by cross-validation
scores=cross_val_score(lr,X,y,cv=5)
print("scores = {} \n final score = {}\n".format(scores,scores.mean()))
print("\n")


