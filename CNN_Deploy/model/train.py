#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:49:26 2020

@author: sundeep1501
"""

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris_df = datasets.load_iris()

x = iris_df.data
y = iris_df.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
dt = DecisionTreeClassifier().fit(X_train, y_train)
preds = dt.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy))
