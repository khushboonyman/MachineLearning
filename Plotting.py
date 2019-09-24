#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:49:53 2019

@author: enrico
"""

from DataPreparation import *

import numpy as np
import matplotlib.pyplot as plt

## Classification problem
# The current variables X and y represent a classification problem, in
# which a machine learning model will use the sepal and petal dimesions
# (stored in the matrix X) to predict the class (species of Iris, stored in
# the variable y). A relevant figure for this classification problem could
# for instance be one that shows how the classes are distributed based on
# two attributes in matrix X:
X_c = X.copy();

y_c = educationVector.copy();

attributeNames_c = attributeNames.copy();
i = 5; j = 6;
color = ['r','b', 'g', 'c', 'm', 'y']


fig0, ax1 = plt.subplots()
ax1.set_title('Education calssification')
for c in range(len(educationNames)):
    idx = y_c == c
    ax1.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=list(educationDict.keys())[list(educationDict.values()).index(c)])
ax1.legend()
ax1.set_xlabel(attributeNames_c[i])
ax1.set_ylabel(attributeNames_c[j])


fig1, ax2 = plt.subplots()
y_c = genderVector.copy()

ax2.set_title('Gender calssification')
for c in range(len(genderNames)):
    idx = y_c == c
    ax2.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=list(genderDict.keys())[list(genderDict.values()).index(c)])
ax2.legend()
ax2.set_xlabel(attributeNames_c[i])
ax2.set_ylabel(attributeNames_c[j])

y_c = educationVector.copy();

# Regression of the gender classification
data = np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
# Get the Reading score

COLOUMN = 6
y_r = data[:, COLOUMN]
x_list = list(range(len(X_c[0]) + 1))
x_list.remove(COLOUMN)

X_r = data[:, x_list]

gender = np.array(X_r[:, -1], dtype=int).T
K = gender.max()+1
gender_encoding = np.zeros((gender.size, K))
gender_encoding[np.arange(gender.size), gender] = 1

X_r = np.concatenate( (X_r[:, :-1], gender_encoding), axis=1) 
targetName_r = attributeNames_c[COLOUMN]

x_list = list(range(len(X_c[0])))
x_list.remove(COLOUMN)
print(list(educationDict.keys()))
print(educationNames)
attributeNames_r = np.concatenate((attributeNames_c[x_list], list(genderDict.keys())), axis=0)
print(attributeNames_r)
N,M = X_r.shape
print(X_r)
fig2, ax3 = plt.subplots()

i = COLOUMN
ax3.set_title('Gender regression problem')
ax3.plot(X_r[:, i], y_r, 'o', c=color[4])
ax3.set_xlabel(attributeNames_r[i]);
ax3.set_ylabel(targetName_r);
plt.show()