# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2 2019

@author: Enrico, Khushboo, Alex
"""
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
# Load csv file with data
doc = pd.read_csv('C:\\Users\\Bruger\\Desktop\\books\\02450_introduction_to_machine_learning_and_data_mining\\02450Toolbox_Python\\Data\\StudentsPerformance.csv')

#encoding columns : parental level of education, lunch, test preparation course 
#parental level of education is ordinal, so we assign rankings to them
parEdu = doc['parental level of education']
classNameParEdu = sorted(set(parEdu))
parEduDict = {'some high school':1,
             'high school':2,
             'some college':3,
             "associate's degree":4,
             "bachelor's degree":5,
             "master's degree":6}
doc['parental level of education'].replace(parEduDict, inplace=True)

#we assume that the poor students get free or subsidised lunch and rich students pay standard price.
#this resulted in ranking lunch to denote ranking of income group of the student's family
lunch = doc['lunch']
classNameLunch = sorted(set(lunch))
lunchDict = {'free/reduced':1,
             'standard':2}
doc['lunch'].replace(lunchDict, inplace=True)

#a student is better prepared when he has completed the test preparation course, else not. So, this 
#attribute can also be ranked
testPrepCourse = doc['test preparation course']
classtestPrepCourse = sorted(set(testPrepCourse))
testPrepCourseDict = {'none':0,
                      'completed':1}
doc['test preparation course'].replace(testPrepCourseDict, inplace=True)

#one in K encoing of columns : gender, race/ethinicity 
#since gender and ethinicity cannot be ranked, so we decided to one in K encode them
doc = pd.get_dummies(doc,prefix=['gender_','race_'],columns=['gender','race/ethnicity'])

# Extract attribute names 
attributeNames = list(doc.columns)[1:]

print('Data preparation done!!')
#we decided to predict math score based on all other attributes
listOfAttribute = list(i for i in range(13) if i != 3)

X = np.asarray(doc.iloc[:,listOfAttribute])
y = np.asarray(doc.iloc[:,3])

attributeNames = list(doc.columns)
attributeNames.remove('math score')
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
#C = len(className)

print('Data preparation for regression problem!!')
#REGRESSION
#standardization

X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))

print('Standardization done!!')

