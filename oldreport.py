# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:41:32 2019

@author: Bruger
"""

#REGRESSION
#standardization
X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))

print('Standardization done!!')

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
internal_cross_validation = 10
CV = model_selection.KFold(K, shuffle=True)

#SETUP FOR REGULARIZATION
# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
lambda_star = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

#SETUP FOR ANN
Error_test_ann = np.empty((K,1))
n_hidden_units = 15   # number of hidden units
n_replicates = 2        # number of networks trained in each k-fold
max_iter = 10000        # 
# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )

loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
print('Training ANN model of type:\n\n{}\n'.format(str(model())))

k=0

def regular():
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for making new predictions)
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    lambda_star[k] = opt_lambda
    
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()  
        
    return y_train,y_test
    
def ANN():
    # Extract training and test set for current CV fold, convert to tensors
    X_train_torch = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train_torch = torch.tensor(y[train_index], dtype=torch.float)
    X_test_torch = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test_torch = torch.tensor(y[test_index], dtype=torch.uint8)
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_torch,
                                                       y=y_train_torch,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test_torch)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test_torch.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test_torch)).data.numpy() #mean
    print(mse)
    Error_test_ann[k] = mse # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    return y_test_est,y_test_torch,net

def displayReg():
    # Display results
    print('Regularized linear regression:')
    print('- Training error: {0}'.format(Error_train_rlr.mean()))
    print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
    
    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

    print('REGULARIZATION WITH',K,internal_cross_validation, 'FOLDS!!!!')
    
def displayANN():
    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(Error_test_ann), color=color_list)
    summaries_axes[1].set_xlabel('Fold');
    summaries_axes[1].set_xticks(np.arange(1, K+1))
    summaries_axes[1].set_ylabel('MSE');
    summaries_axes[1].set_title('Test mean-squared-error')
    
    print('Diagram of best neural net in last fold:')
    weights = [net[i].weight.data.numpy().T for i in [0,2]]
    biases = [net[i].bias.data.numpy() for i in [0,2]]
    tf =  [str(net[i]) for i in [1,2]]
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

    print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(Error_test_ann)), 4)))

    plt.figure(figsize=(10,10));
    y_est = y_test_est.data.numpy(); y_true = y_test_torch.data.numpy();
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

    print('ANN WITH',K,'FOLDS!!!!')

    
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K)) 
    
    y_train, y_test = regular()    
    y_test_est,y_test_torch,net = ANN()
        
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    k+=1

show()

displayReg()

displayANN()