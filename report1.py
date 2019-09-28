# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:21:05 2019

@author: Enrico, Khushboo, Alex
"""

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd
# Load csv file with data
doc = pd.read_csv('C:\\Users\\Bruger\\Desktop\\books\\02450_introduction_to_machine_learning_and_data_mining\\02450Toolbox_Python\\Data\\StudentsPerformance.csv')

doc.drop(columns=['race/ethnicity','parental level of education','lunch','test preparation course'],inplace=True)

# matrices for different scores
mathScore = np.asarray(doc['math score'])
readingScore = np.asarray(doc['reading score'])
writingScore = np.asarray(doc['writing score'])

print("correlation of math score vs reading score : ",  np.corrcoef(mathScore,readingScore))
print("correlation of math score vs writing score : ", np.corrcoef(mathScore,writingScore))
print("correlation of reading score vs writing score : ", np.corrcoef(readingScore,writingScore))

print("covariance of math score vs reading score", np.cov(mathScore,readingScore))
print("covariance of math score vs reading score", np.cov(mathScore, writingScore))
print("covariance of math score vs reading score", np.cov(readingScore, writingScore))
print("covariance of math score",np.cov(mathScore))
print("covariance of reading score",np.cov(readingScore))
print("covariance of reading score",np.cov(writingScore))


plt.matshow(doc.corr())
plt.show()

# Extract attribute names 
attributeNames = list(doc.columns)[1:]

# Extract class names to python list,# then encode with integers (dict)
classLabel = doc['gender']
className = sorted(set(classLabel))

"""className2 = sorted(set(doc['race/ethnicity']))
className3 = sorted(set(doc['parental level of education']))
className4 = sorted(set(doc['lunch']))
className5 = sorted(set(doc['test preparation course']))
doc.describe()"""

classDict = dict(zip(className, range(2)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabel])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((1000, 3))
for i, col_id in enumerate(range(1,4)):
    X[:, i] = np.asarray(doc.iloc[:,col_id])

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(className)

print('End of part 1')

sb.boxplot(mathScore)
sb.boxplot(readingScore)
sb.boxplot(writingScore)

# Two constants
"""a = 1.5
b = 1.5

# Check the statements in the exercise
print("Cosine scaling: %.4f " % (similarity(mathScore,writingScore,'cos') - similarity(a*mathScore,writingScore,'cos'))[0,0])
print("ExtendedJaccard scaling: %.4f " % (similarity(mathScore,y,'ext') - similarity(a*mathScore,writingScore,'ext'))[0,0])
print("Correlation scaling: %.4f " % (similarity(mathScore,writingScore,'cor') - similarity(a*mathScore,writingScore,'cor'))[0,0])
print("Cosine translation: %.4f " % (similarity(mathScore,writingScore,'cos') - similarity(b+mathScore,writingScore,'cos'))[0,0])
print("ExtendedJaccard translation: %.4f " % (similarity(mathScore,writingScore,'ext') - similarity(b+mathScore,writingScore,'ext'))[0,0])
print("Correlation translation: %.4f " % (similarity(mathScore,writingScore,'cor') - similarity(b+mathScore,writingScore,'cor'))[0,0])
"""

#math score vs reading score
i = 0
j = 1

plot(X[:, i], X[:, j], 'o')

f = figure()
title('Student Score')


for c in range(C):
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(className)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

#math score vs writing score
i = 0
j = 2

plot(X[:, i], X[:, j], 'o')

f = figure()
title('Student Score')


for c in range(C):
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(className)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

#reading score vs writing score
i = 1
j = 2

plot(X[:, i], X[:, j], 'o')

f = figure()
title('Student Score')


for c in range(C):
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(className)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

print('End of part 2')

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

#to check 90%
threshold = 0.9

#to check 95%
threshold_2 = 0.99

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.plot([1,len(rho)],[threshold_2, threshold_2],'g-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('End of part 3')

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Student Score : PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(className)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

i = 0
j = 2

# Plot PCA of the data
f = figure()
title('Student Score : PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(className)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

i = 1
j = 2

# Plot PCA of the data
f = figure()
title('Student Score : PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(className)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

print('End of part 4')

pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Student Score: PCA Component Coefficients')
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes A, E and H. We can confirm
# this by looking at it's numerical values directly, too:
print('PC2:')
print(V[:,1].T)

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.
all_female_data = Y[y==0,:]

print('First female observation')
print(all_female_data[0,:])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):
print('...and its projection onto PC2')
print(all_female_data[0,:]@V[:,1])
# Try to explain why?
