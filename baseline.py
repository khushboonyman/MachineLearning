"""
Created on Sat Nov 2 2019
@author: Enrico, Khushboo, Alex
"""
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from data_preparation_n_standarization import *

K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_noreg = np.empty((M,K))
i=0


for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    dy = []
    dy1 = []
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[i] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[i] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]*100
    
    
    i+=1
    


# =============================================================================
#     Xty = X_train.T @ y_train
#     XtX = X_train.T @ X_train
#      # Estimate weights for unregularized linear regression, on entire training set
#     w_noreg[:,i] = np.linalg.solve(XtX,Xty).squeeze()
#     # Compute mean squared error without regularization
#     Error_train[i] = np.square(y_train-X_train @ w_noreg[:,i]).sum(axis=0)/y_train.shape[0]
#     Error_test[i] = np.square(y_test-X_test @ w_noreg[:,i]).sum(axis=0)/y_test.shape[0]
#     
#     i+=1
#     
# # Plot the classification error rate
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
# =============================================================================
