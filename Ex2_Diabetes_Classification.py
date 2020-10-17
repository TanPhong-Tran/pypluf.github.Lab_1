
# # Ex2_Diabetes_Classification


# Load libraries 
import pandas as pd
#Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
#Import train_test_split function
from sklearn.model_selection import train_test_split
#Import metrics for accuracy calculation
from sklearn import metrics
import eli5 #Calculating and Displaying impoetance using the eli5 library
from eli5.sklearn import PermutationImportance



#import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
#Load dataset
df = pd.read_csv("diabetes.csv",header = 0, names = col_names)

df.head()




df.info()




# Split dataset in feature and target variable
feature_cols =  ['pregnant','insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = df[feature_cols] #Features
y = df.label #Target variable


# Slitting dataset into training set and test set
# 70% training set and 30% test set
X_train, X_test ,y_train, y_test= train_test_split(X,y,test_size = 0.3, random_state = 1)


## Decision Tree Classifier

# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the reponse for test dataset
y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy: {}".format(sum(y_pred == y_test) / len(y_pred)))
print("CART (Tree Prediction) Accuracy by calling metrics: ", metrics.accuracy_score(y_test, y_pred))




# Evaluate a score by cross validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))


#  Random Forest Classifier

#Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)
#Predict testset
y_pred = rdf.predict(X_test)
#Evaluate permance of the model
print('RDF: ', metrics.accuracy_score(y_test, y_pred))
print('\n')
#Evaluate a score by cross-validatioabsn
scores = cross_val_score(rdf,X,y,cv=5)
print('scores = {} \n final score = {} \n'.format(scores, scores.mean()))
print('\n')


## Logistic Regression Classifier


#import warning filter
from warnings import simplefilter
#ignore all future warnings
simplefilter(action = 'ignore', category = FutureWarning)

# Fit logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict testset
y_pred = lr.predict(X_test)
#Evaluate performance of the model
print('LR: ', metrics.accuracy_score(y_test, y_pred))
#Evaluate a score by cross-validation
scores = cross_val_score(lr,X,y,cv=5)
print('score = {} \n final score = {} \n'.format(scores, scores.mean()))
print('\n')




