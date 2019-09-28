#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:36:20 2019

@author: enrico
"""

# exercise 1.5.1
import numpy as np
import pandas as pd

filename = 'StudentsPerformance.csv'
df = pd.read_csv(filename)

raw_data = np.asarray(df)

cols = range(0, 8) 
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

genderLabels = raw_data[:,0] # -1 takes the last column
genderNames = np.unique(genderLabels)
genderDict = dict(zip(genderNames,range(len(genderNames))))
genderVector = np.array([genderDict[cl] for cl in genderLabels])


educationDict =	{
  "some high school": 0,
  "high school": 1,
  "some college": 2,
  "associate's degree": 3,
  "bachelor's degree": 4,
  "master's degree": 5
}

educationLabels = raw_data[:,2]
educationNames = np.unique(educationLabels)
educationVector = np.array([educationDict[cl] for cl in educationLabels])


lunchLabel = raw_data[:,3]
lunchNames = np.unique(lunchLabel)
lunchDict = dict(zip(lunchNames,range(len(lunchNames))))
lunchVector = np.array([lunchDict[cl] for cl in lunchLabel])

prepLabel = raw_data[:,4]
prepNames = np.unique(prepLabel)
prepDict = dict(zip(prepNames,range(len(prepNames))))
prepVector = np.array([prepDict[cl] for cl in prepLabel])

N, M = X.shape

