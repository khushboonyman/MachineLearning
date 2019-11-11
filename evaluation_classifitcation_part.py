import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from data_preparation_n_standarization import *
import warnings
warnings.filterwarnings("ignore")
import scipy.stats
import numpy as np
import scipy.stats as st

def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1) * (Q-1)
    q = (1-Etheta) * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p




yhat = []
y_true = []
final_opt_lambda = 0.2223
final_opt_K = 2
for train_index, test_index in CV.split(X, y):  
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    dy = []
    dy1 = []
    
    # Train KNN method with optimal lambda
    knclassifier = KNeighborsClassifier(n_neighbors=int(final_opt_K));
    dummyclassifier = DummyClassifier()
    logisticclassifier = LogisticRegression(penalty='l2', C=1/final_opt_lambda )
    
    knclassifier.fit(X, y);
    dummyclassifier.fit(X, y);
    logisticclassifier.fit(X, y);
    
    y_est_knn = knclassifier.predict(X_test);
    y_est_dummy = dummyclassifier.predict(X_test);
    y_est_logistic = logisticclassifier.predict(X_test);
    
    dy.append( y_est_knn )
    dy.append( y_est_dummy )
    dy.append( y_est_logistic )
    
    dy1 = np.stack(dy, axis=1)
    yhat.append(dy1)
    y_true.append(y_test)

    

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)

alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat2, CI2, p2] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat2, " CI: ", CI2, "p-value", p2)
 
[thetahat3, CI3, p3] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat3, " CI: ", CI3, "p-value", p3)

