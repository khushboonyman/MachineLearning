#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:49:53 2019

@author: enrico
"""

from DataPreparation import *

import numpy as np
import matplotlib.pyplot as plt


def getIndex(a, value):
    x = np.where(a == value)
    return x[0][0]


# copy the full matrix of data
X_c = X.copy()

edv_c = educationVector.copy()

attributeNames_c = attributeNames.copy()

i = getIndex(attributeNames_c, 'math score')
j = getIndex(attributeNames_c, 'reading score')

color = ['r','b', 'g', 'c', 'm', 'y']

color_hex = ['#6b4c9a', '#cc5e60', '#da7c30', '#958b3d']

fig0, ax1 = plt.subplots()
ax1.set_title('Education calssification based on parent education')
for c in range(len(educationNames)):
    idx = edv_c == c
    ax1.scatter(x=X_c[idx, i],
                y=X_c[idx, j],
                c=color[c],
                s=50, alpha=0.5,
                label=list(educationDict.keys())[list(educationDict.values()).index(c)])
ax1.legend()
ax1.set_xlabel(attributeNames_c[i])
ax1.set_ylabel(attributeNames_c[j])


fig1, ax2 = plt.subplots()
gv_c = genderVector.copy()

ax2.set_title('Education calssification based on gender')
for c in range(len(genderNames)):
    idx = gv_c == c
    ax2.scatter(x=X_c[idx, i],
                y=X_c[idx, j],
                c=color[c],
                s=50, alpha=0.5,
                label=list(genderDict.keys())[list(genderDict.values()).index(c)])
ax2.legend()
ax2.set_xlabel(attributeNames_c[i])
ax2.set_ylabel(attributeNames_c[j])

lunv_c = lunchVector.copy()

fig3, ax4 = plt.subplots()
ax4.set_title('Education calssification based on social status')
for c in range(len(lunchNames)):
    idx = lunv_c == c
    ax4.scatter(x=X_c[idx, i],
                y=X_c[idx, j],
                c=color[c+3],
                s=50, alpha=0.5,
                label=list(lunchDict.keys())[list(lunchDict.values()).index(c)])
ax4.legend()
ax4.set_xlabel(attributeNames_c[i])
ax4.set_ylabel(attributeNames_c[j])


prepv_c = prepVector.copy()

fig4, ax5 = plt.subplots()
ax5.set_title('Education calssification based preparation')
for c in range(len(prepNames)):
    idx = prepv_c == c
    ax5.scatter(x=X_c[idx, i],
                y=X_c[idx, j],
                c=color_hex[c+2],
                s=50, alpha=0.5,
                label=list(prepDict.keys())[list(prepDict.values()).index(c)])
ax5.legend()
ax5.set_xlabel(attributeNames_c[i])
ax5.set_ylabel(attributeNames_c[j])

COLUMN = getIndex(attributeNames_c,"reading score")
# Regression of the gender classification

# concatenate the gender vector to the matrix in order to do a one-out-of-K encoding 
data = np.concatenate((X_c, np.expand_dims(gv_c,axis=1)), axis=1)

# Save the reading score in to a new vector and remove it from the global data
y_r = data[:, COLUMN]
new_attr_vec = list(range(len(X_c[0]) + 1))
new_attr_vec.remove(COLUMN)

X_r = data[:, new_attr_vec]

gender = np.array(X_r[:, -1], dtype=int).T
K = gender.max()+1
gender_encoding = np.zeros((gender.size, K))
gender_encoding[np.arange(gender.size), gender] = 1

X_r = np.concatenate( (X_r[:, :-1], gender_encoding), axis=1)
targetName_r = attributeNames_c[COLUMN]

new_attr_vec = list(range(len(X_c[0])))
new_attr_vec.remove(COLUMN)
attributeNames_r = np.concatenate((attributeNames_c[new_attr_vec], list(genderDict.keys())), axis=0)
N,M = X_r.shape
fig2, ax3 = plt.subplots()

i = COLUMN
ax3.set_title('Gender regression problem')
ax3.plot(X_r[:, i], y_r, 'o', c=color_hex[0], alpha=0.5)
ax3.set_xlabel(attributeNames_r[i]);
ax3.set_ylabel(targetName_r);
plt.show()