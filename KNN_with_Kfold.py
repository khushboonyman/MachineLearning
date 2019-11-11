"""
Created on Sat Nov 2 2019
@author: Enrico, Khushboo, Alex
"""
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, scatter
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
L= np.arange(1, 41, 1)

K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

Error_test = np.empty((K,1))
final_opt_lambdas = np.empty((K,1))
i=0
dy = []
yhat = []
y_true = []
for train_index, test_index in CV.split(X, y):  
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    final_min_test_error, final_opt_lambda = find_optimal_lambda_for_knn(X_train, y_train, L, K)
    
    Error_test[i] = final_min_test_error
    final_opt_lambdas[i] = final_opt_lambda
        
    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))  
    i+=1
    
min_error = np.min(Error_test)
final_opt_lambda_idx = np.argmin(final_opt_lambdas)
final_opt_lambda = final_opt_lambdas[final_opt_lambda_idx]

figure()
plot(100*Error_test/N)
#scatter(3, 9, s=10, color="red")
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
print('Min error: {0}'.format(min_error))

