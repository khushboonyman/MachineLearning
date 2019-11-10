import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def find_optimal_lambda(X, y, lambda_interval, cvf):
    CV = model_selection.KFold(n_splits=cvf,shuffle=True)    
    opt_lambdas = np.empty((cvf,1))
    min_test_errors = np.empty((cvf,1))
    min_train_errors = np.empty((cvf,1))
    
    k=0
    for train_index, test_index in CV.split(X):

        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        coefficient_norm = np.zeros(len(lambda_interval))
        for l in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[l] )
            mdl.fit(X_train, y_train)
        
            y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_test).T
            
            train_error_rate[l] = np.sum(y_train_est != y_train) / len(y_train)
            test_error_rate[l] = np.sum(y_test_est != y_test) / len(y_test)
        
            w_est = mdl.coef_[0] 
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
        
        min_test_errors[k] = np.min(test_error_rate)
        min_train_errors[k] = np.min(train_error_rate)
        
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        opt_lambdas[k] = opt_lambda
        
        k+=1
        
    final_min_test_error = np.min(min_test_errors)
    opt_lambda_idx_2 = np.argmin(min_test_errors)
    final_opt_lambda = opt_lambdas[opt_lambda_idx_2]
    
    final_min_train_error = np.min(min_train_errors)
    
# =============================================================================
#     plt.figure(figsize=(8,8))
#     plt.semilogx(lambda_interval, train_error_rate*100)
#     plt.semilogx(lambda_interval, test_error_rate*100)
#     plt.semilogx(final_opt_lambda, final_min_test_error*100, 'o')
#     plt.text(1e-8, 3, "Minimum test error: " + str(np.round(final_min_test_error*100,2)) + ' % at 1e' + str(np.round(np.log10(final_opt_lambda),2)))
#     plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
#     plt.ylabel('Error rate (%)')
#     plt.title('Classification error')
#     plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
#     plt.ylim([0, 20])
#     plt.grid()
#     plt.show()    
#     
#     plt.figure(figsize=(8,8))
#     plt.semilogx(lambda_interval, coefficient_norm,'k')
#     plt.ylabel('L2 Norm')
#     plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
#     plt.title('Parameter vector L2 norm')
#     plt.grid()
#     plt.show()    
# =============================================================================

    print('Ran Find Optimal Lambda')

    return final_min_test_error, final_min_train_error, final_opt_lambda


def find_optimal_lambda_for_knn(X, y, L, cvf):
    CV = model_selection.KFold(n_splits=cvf,shuffle=True)    
    opt_lambdas = np.empty((cvf,1))
    min_test_errors = np.empty((cvf,1))
    min_train_errors = np.empty((cvf,1))
    

    errors = np.zeros(len(L))
    i=0
    for train_index, test_index in CV.split(X, y):
        #print('Crossvalidation fold: {0}/{1}'.format(i+1,cvf))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,len(L)+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            cm = confusion_matrix(y_test, y_est);
            accuracy = 100*cm.diagonal().sum()/cm.sum()
            errors[l-1] = 100-accuracy
            #errors[l-1] = np.sum(y_est[0]!=y_test[0])
    
        
        min_test_errors[i] = np.min(errors)
        opt_lambda_idx = np.argmin(errors)
        opt_lambdas[i] = L[opt_lambda_idx]
        
        i+=1
        
    # Plot the classification error rate
# =============================================================================
#     figure()
#     plot(100*sum(errors,0)/N)
#     xlabel('Number of neighbors')
#     ylabel('Classification error rate (%)')
#     show()
# =============================================================================
    
        
    final_min_test_error = np.min(min_test_errors)
    opt_lambda_idx_2 = np.argmin(min_test_errors)
    final_opt_lambda = opt_lambdas[opt_lambda_idx_2]
    
    return final_min_test_error, final_opt_lambda