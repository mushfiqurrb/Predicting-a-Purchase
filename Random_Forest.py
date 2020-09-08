# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:21:29 2020

@author: m_rah
"""

# Loading the Dataset
import pandas as pd
dataset = pd.read_csv("phone_purchase_records.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Converting Gender into Number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
x[:,0] = labelEncoder_gender.fit_transform(x[:,0])

# Spliting Data into Training and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Fitting Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
classifier.fit(x_train, y_train)

# Making Predictions
y_pred = classifier.predict(x_test)

# Evaluate Performance of the Model
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred) 
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred) 
print("Recall score:",recall)