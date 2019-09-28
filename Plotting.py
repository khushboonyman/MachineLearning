#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:49:53 2019

@author: enrico
"""

from DataPreparation import *
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib.mlab as mlab

from scipy.stats import linregress
from similarity import similarity

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

marker = ['.', 'x', 'd', 'h']
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
                marker = marker[c],
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
#data = np.concatenate((X_c, np.expand_dims(gv_c,axis=1)), axis=1)
#
## Save the reading score in to a new vector and remove it from the global data
#y_r = data[:, COLUMN]
#new_attr_vec = list(range(len(X_c[0]) + 1))
#new_attr_vec.remove(COLUMN)
#
#X_r = data[:, new_attr_vec]
#
#gender = np.array(X_r[:, -1], dtype=int).T
#K = gender.max()+1
#gender_encoding = np.zeros((gender.size, K))
#gender_encoding[np.arange(gender.size), gender] = 1
#
#X_r = np.concatenate( (X_r[:, :-1], gender_encoding), axis=1)
#targetName_r = attributeNames_c[COLUMN]
#
#new_attr_vec = list(range(len(X_c[0])))
#new_attr_vec.remove(COLUMN)
#attributeNames_r = np.concatenate((attributeNames_c[new_attr_vec], list(genderDict.keys())), axis=0)
#N,M = X_r.shape
#fig2, ax3 = plt.subplots()
#
#i = COLUMN
#ax3.set_title('Gender regression problem')
#ax3.plot(X_r[:, i], y_r, 'o', c=color_hex[0], alpha=0.5)
#ax3.set_xlabel(attributeNames_r[i]);
#ax3.set_ylabel(targetName_r);
##plt.show()

Mt = X.copy()
Mt = np.asanyarray(Mt[:, 5:])
Mt = Mt.astype(None)

Y = Mt - np.ones((N,1))* Mt.mean(axis=0)
mean_x = np.mean(Mt[0,:])
mean_y = np.mean(Mt[1,:])
U,S,Vt = svd(Y,full_matrices=False)
V = Vt.T
slope, intercept, r_value, p_value, std_err = linregress(V[:,0], V[:,1])
eig_pairs = [(np.abs(S[i]), U[:,i]) for i in range(len(S))]

eig_pairs.sort()
eig_pairs.reverse()


matrix_w = np.hstack((eig_pairs[0][1].reshape(1000,1),
                      eig_pairs[1][1].reshape(1000,1)))



fig5, ax6 = plt.subplots()
print(1000*V[0][0])
print(mean_x, mean_y)
ax6.arrow(mean_x, mean_y, 200 * V[0][0], 200 * V[0][1], head_width=2, head_length=2, fc='k', ec='k')
ax6.arrow(mean_x, mean_y, 200 * V[1][0], 200 * V[1][1], head_width=2, head_length=2, fc='k', ec='k')

#ax6.arrow(0, 0, -200 * V[0][0], -200 * V[0][1], head_width=2, head_length=2, fc='k', ec='k')

#ax6.arrow(mean_x, mean_y, 20 * V[1][0], 20 * V[1][1], head_width=2, head_length=2, fc='k', ec='k')

ax6.set_xlim([0,100])
ax6.set_ylim([0,100])
ax6.set_title('PCA1 and PCA2 directions')
for c in range(len(genderNames)):
    idx = gv_c == c
    ax6.scatter(x=X_c[idx, i],
                y=X_c[idx, j],
                c=color[c],
                s=50, alpha=0.1,
                label=list(genderDict.keys())[list(genderDict.values()).index(c)])
ax6.legend()
ax6.set_xlabel(attributeNames_c[i])
ax6.set_ylabel(attributeNames_c[j])
ax6.legend()
plt.show()




math_score = np.asarray(df['math score'])
reading_score = np.asarray(df['reading score'])
writing_score = np.asarray(df['writing score'])

mu = np.mean(reading_score)
fig6, ax7 = plt.subplots()
ax7.set_title('Histogram reading score')
#ax7.hist(reading_score, 10, normed=1, facecolor='#2f7ed8', alpha=0.3)
#y = mlab.normpdf(10, np.mean(reading_score), np.std(reading_score))
#y = ((1 / (np.sqrt(2 * np.pi) * np.std(reading_score))) *
#     np.exp(-0.5 * (1 / np.std(reading_score) * (10 - np.mean(reading_score)))**2))
#ax7.plot(10, y, 'r--')
plt.figure()
sns.distplot(reading_score, hist=True, kde=True, 
             bins=int(100/10), color = 'lightBlue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})
plt.title("Histogram reading score")
plt.figure()
sns.distplot(math_score, hist=True, kde=True, 
             bins=int(100/10), color = 'lightGreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})
plt.title("Histogram math score")
plt.figure()
wsm = np.mean(writing_score)
ws_m = writing_score - wsm
sns.distplot(writing_score, hist=True, kde=True, 
             bins=int(100/10), color = 'red', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})
plt.title("Histogram writing score")



sim = similarity(writing_score, reading_score, 'cor')
print(sim)

score_attribute = attributeNames[5:8]
print(score_attribute)
i = 0
j = 1
fig7, ax8 = plt.subplots()
for att in range(3):
    ax8.arrow(0,0, V[att,i], V[att,j])
    ax8.text(V[att,i], V[att,j], score_attribute[att])
ax8.set_xlim([-1,1])
ax8.set_ylim([-1,1])
ax8.set_xlabel('PC'+str(i+1))
ax8.set_ylabel('PC'+str(j+1))
ax8.grid()
    # Add a unit circle
ax8.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
#21ax8.title(titles[k] +'\n'+'Attribute coefficients')
ax8.axis('equal')

rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# %%
score_frame = df[['math score', 'reading score', 'writing score']]
plt.figure()
sns.boxplot(data=score_frame, palette="Set3")

# %% 
