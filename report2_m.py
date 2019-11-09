# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2 2019

@author: Enrico, Khushboo, Alex
"""
import pandas as pd
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import matplotlib.pyplot as plt
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm

# Load csv file with data
doc = pd.read_csv('~/Git/MachineLearning/StudentsPerformance.csv')

#encoding columns : parental level of education, lunch, test preparation course
#parental level of education is ordinal, so we assign rankings to them
parEdu = doc['parental level of education']
classNameParEdu = sorted(set(parEdu))
parEduDict = {'some high school':1,
             'high school':2,
             'some college':3,
             "associate's degree":4,
             "bachelor's degree":5,
             "master's degree":6}
doc['parental level of education'].replace(parEduDict, inplace=True)

#we assume that the poor students get free or subsidised lunch and rich students pay standard price.
#this resulted in ranking lunch to denote ranking of income group of the student's family
lunch = doc['lunch']
classNameLunch = sorted(set(lunch))
lunchDict = {'free/reduced':1,
             'standard':2}
doc['lunch'].replace(lunchDict, inplace=True)

#a student is better prepared when he has completed the test preparation course, else not. So, this
#attribute can also be ranked
testPrepCourse = doc['test preparation course']
classtestPrepCourse = sorted(set(testPrepCourse))
testPrepCourseDict = {'none':0,
                      'completed':1}
doc['test preparation course'].replace(testPrepCourseDict, inplace=True)

#one in K encoing of columns : gender, race/ethinicity
#since gender and ethinicity cannot be ranked, so we decided to one in K encode them
doc = pd.get_dummies(doc,prefix=['gender_','race_'],columns=['gender','race/ethnicity'])

# Extract attribute names
attributeNames = list(doc.columns)[1:]

print('Data preparation done!!')
#we decided to predict math score based on all other attributes
listOfAttribute = list(i for i in range(13) if i != 3)

X = np.asarray(doc.iloc[:,listOfAttribute])
y = np.asarray(doc.iloc[:,3])
y.shape = (len(y),1)

attributeNames = list(doc.columns)
attributeNames.remove('math score')
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
#C = len(className)

print('Data preparation for regression problem!!')
#REGRESSION
#standardization

#%%
K = 12
k = 5
cvf = 10
kf = KFold(n_splits=k)
kf.get_n_splits(X)
lambdas = np.power(10.,range(-5,9))


CV = model_selection.KFold(cvf, shuffle=True)
M = X.shape[1]
w = np.empty((M,cvf,len(lambdas)))
train_error = np.empty((cvf,len(lambdas)))
test_error = np.empty((cvf,len(lambdas)))
f = 0
y = y.squeeze()
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

    f=f+1

opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))


figure(k, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()


print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w[m,-1, np.argmin(np.mean(test_error,axis=0))],2)))

