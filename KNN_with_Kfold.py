"""
Created on Sat Nov 2 2019
@author: Enrico, Khushboo, Alex
"""
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from data_preparation_n_standarization import *
from ml_utils import find_optimal_lambda_for_knn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# requires data from exercise 1.5.1
#from ex1_5_1 import *

# Maximum number of neighbors
L= np.arange(40)

K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

Error_test = np.empty((K,1))
opt_lambdas = np.empty((K,1))
i=0
for train_index, test_index in CV.split(X, y):  
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    final_min_test_error, final_opt_lambda = find_optimal_lambda_for_knn(X_train, y_train, L, K)
    
    Error_test[i] = final_min_test_error
    opt_lambdas[i] = final_opt_lambda
    
    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))  
    i+=1
    
min_error = np.min(Error_test)
opt_lambda_idx = np.argmin(opt_lambdas)
opt_lambda = opt_lambdas[opt_lambda_idx]

plt.figure(figsize=(8,8))
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
