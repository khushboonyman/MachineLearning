# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:21:42 2019

@author: Bruger
"""
#logistic regression and baseline without k fold 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
#import datapreparationclassification
# Create crossvalidation partition for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05, stratify=y)

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

mdl = LogisticRegression(penalty='l2')
    
mdl.fit(X_train, y_train)
y_train_est = mdl.predict(X_train).T
y_test_est = mdl.predict(X_test).T
# Fit logistic regression model to training data to predict the gender

Error_train_nofeatures = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
Error_test_nofeatures= np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

w_est = mdl.coef_[0] 
coefficient_norm = np.sqrt(np.sum(w_est**2))
