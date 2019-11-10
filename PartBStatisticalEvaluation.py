# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 11:18:39 2019

@author: Bruger
"""
#from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
#import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import model_selection
#import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st
from toolbox_02450 import mcnemar

# requires data from exercise 1.5.1
from regressionPartBANN import *
alpha = 0.05

y_test_acc = np.asarray(y_test_acc)
y_test_ann = list(float(i) for i in y_test_ann)
y_test_ann = np.asarray(y_test_ann)
y_test_base = np.asarray(y_test_base)
y_test_lin = np.asarray(y_test_lin)


[thetahat, CI, p] = mcnemar(y_test_acc, y_test_ann, y_test_base, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat, CI, p] = mcnemar(y_test_acc, y_test_ann, y_test_lin, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat, CI, p] = mcnemar(y_test_acc, y_test_lin, y_test_base, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)