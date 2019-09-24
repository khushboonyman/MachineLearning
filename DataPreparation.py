#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:36:20 2019

@author: enrico
"""

# exercise 1.5.1
import numpy as np
import pandas as pd

# Load the Iris csv data using the Pandas library
filename = 'StudentsPerformance.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = np.asarray(df)

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 8) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])
# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
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


lunchLabel = raw_data[:,3] # -1 takes the last column
lunchNames = np.unique(lunchLabel)
lunchDict = dict(zip(lunchNames,range(len(lunchNames))))
lunchVector = np.array([lunchDict[cl] for cl in lunchLabel])

prepLabel = raw_data[:,4]
prepNames = np.unique(prepLabel)
prepDict = dict(zip(prepNames,range(len(prepNames))))
prepVector = np.array([prepDict[cl] for cl in prepLabel])

N, M = X.shape

