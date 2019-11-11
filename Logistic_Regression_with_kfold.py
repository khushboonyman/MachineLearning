"""
Created on Sat Nov 2 2019
@author: Enrico, Khushboo, Alex
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from data_preparation_n_standarization import *
from ml_utils import find_optimal_lambda
import warnings
warnings.filterwarnings("ignore")

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
opt_lambdas = np.empty((K,1))
w_rlr = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    lambda_interval = np.logspace(-8, 2, 50)
    
    final_min_test_error, final_opt_lambda = find_optimal_lambda(X_train, y_train, lambda_interval, cvf=10)

    Error_test[k] = final_min_test_error
    opt_lambdas[k] = final_opt_lambda
    
    print('Cross validation fold {0}/{1}'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}'.format(test_index))

    k+=1

min_error = np.min(Error_test)
opt_lambda_idx = np.argmin(min_error)
opt_lambda = opt_lambdas[opt_lambda_idx]

opt_lambdas = np.sort(opt_lambdas, axis=None)
Error_train = np.sort(Error_train, axis=None)
Error_test = np.sort(Error_test, axis=None)


plt.figure(figsize=(8,8))
plt.semilogx(opt_lambdas, Error_train*100)
plt.semilogx(opt_lambdas, Error_test*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 20])
plt.grid()
plt.show()    

# =============================================================================
# plt.figure(figsize=(8,8))
# plt.semilogx(lambda_interval, coefficient_norm,'k')
# plt.ylabel('L2 Norm')
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.title('Parameter vector L2 norm')
# plt.grid()
# plt.show()    
# =============================================================================

print('Ran Logistic Regression')

