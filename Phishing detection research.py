# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 23:06:24 2022

@author:  Yosi gottlieb
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier
from sklearn.model_selection import train_test_split
import joblib

# Reading data from csv file
data = pd.read_csv(r"C:\Users\Yair\Desktop\Yosi\exercise\urldata.csv")
#data = pd.read_csv(r"C:\Users\Yair\Desktop\Yosi\exercise\dataset_small.csv")
data.head()

# Labels
y = data["label"]

# Features
url_list = data["url"]

# Using Tokenizer
vectorizer = CountVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)

# Split into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Model Building using logistic regression
LogReg_model = LogisticRegression()
LogReg_model.fit(X_train, y_train)

# Model Building using DecisionTree
DecisionTree_model = DecisionTreeClassifier()
DecisionTree_model.fit(X_train, y_train)

# Model Building using AdaBoost
AdaBoost_model = AdaBoostClassifier()
AdaBoost_model.fit(X_train, y_train)

# Accuracy of each Model
print("Accuracy of LogisticRegression model is: ",LogReg_model.score(X_test, y_test))
print("Accuracy of DecisionTree model is: ",DecisionTree_model.score(X_test, y_test))
print("Accuracy of AdaBoost model is: ",AdaBoost_model.score(X_test, y_test))

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(DecisionTree_model, filename)

# load the model from disk
loaded_model = joblib.load(filename)

# check URL
URL='https://www.cambridge.org/core/services/aop-cambridge-core/content/view/6780B4B95012B76F33F74DAACE927E22/S106279870300005Xa.pdf/div-class-title-the-myth-of-early-globalization-the-atlantic-economy-1500-1800-div.pdf'
URL_array =vectorizer.transform([URL]).toarray()

result = loaded_model.predict(URL_array)
print(result)
