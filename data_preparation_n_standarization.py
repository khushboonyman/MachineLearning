import pandas as pd
import numpy as np
# Load csv file with data
doc = pd.read_csv('./StudentsPerformance.csv')

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

# subtitude also gender with numbers
gender = doc['gender']
classNameGender = sorted(set(gender))
testGender = {'male':0,
              'female':1}
doc['gender'].replace(testGender, inplace=True)

#one in K encoding of columns : race/ethinicity 
#since ethinicity cannot be ranked, so we decided to one in K encode them
doc = pd.get_dummies(doc,prefix=['race_'],columns=['race/ethnicity'])
# =============================================================================
# race = doc['race/ethnicity']
# classNameRace = sorted(set(race))
# classNameRace = {'group A':0,
#               'group B':1,
#               'group C':2,
#               'group D':3,
#               'group E':4}
# doc['race/ethnicity'].replace(classNameRace, inplace=True)
# =============================================================================


# Extract attribute names 
attributeNames = list(doc.columns)[1:]

print('Data preparation done!!')
#-----------------------------------------------------------------------------------------------------#

#we decided to predict gender based on all other attributes
listOfAttribute = list(i for i in range(len(attributeNames)+1) if i != 0)
X = np.asarray(doc.iloc[:,listOfAttribute])
y = np.asarray(doc.iloc[:,0])

attributeNames = list(doc.columns)
attributeNames.remove('gender')
#attributeNames = attributeNames[5:]
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNameGender)
print('Data preparation for classification problem!!')

#REGRESSION
#standardization
X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))
print('Standardization done!!')
