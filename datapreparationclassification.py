# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:34:35 2019

@author: Bruger
"""

import pandas as pd
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import matplotlib.pyplot as plt
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

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

#encoding the categorical variable gender 
genderDict = {'male':0,
              'female':1}
doc['gender'].replace(genderDict, inplace=True)

#one in K encoding of columns : gender, race/ethinicity 
#since gender and ethinicity cannot be ranked, so we decided to one in K encode them
doc = pd.get_dummies(doc,prefix=['race_'],columns=['race/ethnicity'])

# Extract attribute names 
attributeNames = list(doc.columns)[1:]

X = np.asarray(doc.loc[:,attributeNames])
y = np.asarray(doc.iloc[:,0])
y.shape = (len(y),1)
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)

print('Data preparation for classification done!!')