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
length = len(y_test_acc)
y_test_acc = np.array(y_test_acc).reshape(length,1)
y_test_ann = list(float(i) for i in y_test_ann)
y_test_ann = np.array(y_test_ann).reshape(length,1)
y_test_base = np.array(y_test_base).reshape(length,1)
y_test_lin = np.array(y_test_lin).reshape(length,1)

# perform statistical comparison of the models
# compute z with squared error.
zANN = np.abs(y_test_acc - y_test_ann) ** 2
# compute confidence interval of ANN model 
CIANN = st.t.interval(1-alpha, df=len(zANN)-1, loc=np.mean(zANN), scale=st.sem(zANN))  # Confidence interval

# compute z with squared error.
zLIN = np.abs(y_test_acc - y_test_lin) ** 2
# compute confidence interval of linear regression model
CILIN = st.t.interval(1-alpha, df=len(zLIN)-1, loc=np.mean(zLIN), scale=st.sem(zLIN))  # Confidence interval

# compute z with squared error.
zBASE = np.abs(y_test_acc - y_test_base) ** 2
# compute confidence interval of baseline model 
CIBASE = st.t.interval(1-alpha, df=len(zBASE)-1, loc=np.mean(zBASE), scale=st.sem(zBASE))  # Confidence interval

#compare ANN vs linear regression
zANLI = zANN - zLIN
CIANLI = st.t.interval(1-alpha, len(zANLI)-1, loc=np.mean(zANLI), scale=st.sem(zANLI))  # Confidence interval
p_anli = st.t.cdf( -np.abs( np.mean(zANLI) )/st.sem(zANLI), df=len(zANLI)-1)  # p-value

#compare ANN vs baseline
zBAAN = zBASE - zANN
CIBAAN = st.t.interval(1-alpha, len(zBAAN)-1, loc=np.mean(zBAAN), scale=st.sem(zBAAN))  # Confidence interval
p_baan = st.t.cdf( -np.abs( np.mean(zBAAN) )/st.sem(zBAAN), df=len(zBAAN)-1)  # p-value

#compare baseline vs linear regression
zBALI = zBASE - zLIN
CIBALI = st.t.interval(1-alpha, len(zBALI)-1, loc=np.mean(zBALI), scale=st.sem(zBALI))  # Confidence interval
p_bali = st.t.cdf( -np.abs( np.mean(zBALI) )/st.sem(zBALI), df=len(zBALI)-1)  # p-value