# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:47:05 2019

@author: Bruger
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from regressionPartBLambda import *

#FUNCTION to do internal cross validation for ANN
def ann_validate(X,y,hidden_units,cvf,n_replicates,max_iter):
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    test_error = np.empty((cvf,len(hidden_units)))
    f = 0
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for l in range(0,len(hidden_units)):
            model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, hidden_units[l]), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(hidden_units[l], 1), 
                    # no final tranfer function, i.e. "linear output"
                    )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)    
            y_test_est = net(X_test)
            #print("y estimated",y_test_est)
            #print("y test",y_test)
            # Evaluate test performance
            se = (y_test_est.float()-y_test.float())**2 # squared error
            err = (sum(se).type(torch.float)/len(y_test)).data.numpy()
            test_error[f,l] = err
            optimal_hidden = hidden_units[np.argmin(np.mean(test_error,axis=0))]
        f=f+1
    return optimal_hidden,test_error

X = np.asarray(doc.iloc[:,listOfAttribute])
y = np.asarray(doc.iloc[:,3])
y.shape = (len(y),1)
N, M = X.shape

# Normalize data
X = stats.zscore(X);

# Parameters for neural network classifier
hidden_units = [16,17]
n_hidden_units = 2      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000        

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

Error_test_ann = [] # make a list for storing test ANN error in each loop
optimal_hid_list = [] # make a list of storing optimal hidden units in each loop
for k, interval in enumerate(train_test_index): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    train_index = interval[0]
    test_index = interval[1]
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.uint8)
    
    #INTERNAL CROSS VALIDATION FOR NEURAL NETWORK
    optimal_hidden,test_error = ann_validate(X_train,y_train,hidden_units,internal_cross_validation,n_replicates,max_iter)
    # Define the model
    print("k",k,"   ",optimal_hidden)
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, optimal_hidden), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(optimal_hidden, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    print('Training model of type:\n\n{}\n'.format(str(model())))

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    Error_test_ann.append(mse) # store error rate for current CV fold 
    optimal_hid_list.append(optimal_hidden)
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(Error_test_ann)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE');
summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(Error_test_ann)), 4)))

plt.figure(figsize=(10,10));
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy();
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Math score: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

print('Ran ANN with',K,'X',internal_cross_validation,'fold')