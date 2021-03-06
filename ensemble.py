import h5py
import math
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


emotiveNames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2','lead_brake','gas','brake']
emotiveNumbers= [3,4,6,8,12,14,16,22,24,32,43,51,58,60,63,67,68]

window = 100

totalBreaks = 227
xName = 'P7'
yName = 'P8'

allVals = []

print('reading file')

with h5py.File('VPae.mat', 'r') as f:

    breakVals = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-3]:
        allVals.append(f['cnt']['x'][dataSet][()])

    allVals.append(f['cnt']['x'][emotiveNumbers[emotiveNames.index('gas')]][()])
    allVals.append(f['cnt']['x'][emotiveNumbers[emotiveNames.index('brake')]][()])


print('finished reading file')


label_names = ['Braking','Normal']

feature_names = []
for name in emotiveNames[0:len(emotiveNumbers)-3]:
    for i in range(window):
        feature_names.append(name + str(i))

for i in range(window):
    feature_names.append(emotiveNames[len(emotiveNumbers)-2] + str(i))
for i in range(window):
    feature_names.append(emotiveNames[len(emotiveNumbers)-1] + str(i))

features = []
labels = []

isBraking = False
brakingCounter = 0
barkingNum = 0

for i in range(len(breakVals)):

    if breakVals[i]>0.1 and not isBraking:
        isBraking = True

        feature = []
        for j in range(len(allVals)):
            for k in range(window):
                feature.append(allVals[j][i+k])
        
        features.append(feature)
        labels.append(0)
    
    if brakingCounter==600:
        isBraking = False
        brakingCounter = 0
       
        feature = []
        for j in range(len(allVals)):
            for k in range(window):
                feature.append(allVals[j][i+k])
        
        features.append(feature)
        labels.append(1)
        
        barkingNum+=1
    
    if isBraking:
        brakingCounter+=1

print('finished preping data')

sc = StandardScaler()
features = sc.fit_transform(features)

train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

clf1 = svm.SVC(probability=True)
clf2 = svm.LinearSVC()
lda = LDA()
kneigh = KNeighborsClassifier()
lr = LogisticRegression(max_iter=300)

voting = VotingClassifier(estimators=[('svc', clf1), ('lda', lda), ('kneigh', kneigh), ('lr', lr)], voting='soft')

ensemble1 = RandomForestClassifier(n_estimators=1000, bootstrap = True, max_features =None)
ensemble2 = ExtraTreesClassifier(n_estimators=1000, bootstrap = True, max_features =None)
ensemble3 = AdaBoostClassifier(n_estimators=1000)



# Train our classifier
model2 = clf1.fit(train, train_labels)
model3 = clf2.fit(train, train_labels)
model4 = lda.fit(train, train_labels)
model5 = kneigh.fit(train, train_labels)
model7 = lr.fit(train, train_labels)

modelVoting = voting.fit(train, train_labels)

modelEnsemble1 = ensemble1.fit(train, train_labels)
modelEnsemble2 = ensemble2.fit(train, train_labels)
modelEnsemble3 = ensemble3.fit(train, train_labels)
modelEnsemble4 = ensemble4.fit(train, train_labels)
modelEnsemble5 = ensemble5.fit(train, train_labels)


# Make predictions
preds2 = clf1.predict(test)
preds3 = clf2.predict(test)
preds4 = lda.predict(test)
preds5 = kneigh.predict(test)
preds7 = lr.predict(test)

predVoting = voting.predict(test)

predEnsemble1 = ensemble1.predict(test)
predEnsemble2 = ensemble2.predict(test)
predEnsemble3 = ensemble3.predict(test)
predEnsemble4 = ensemble4.predict(test)
predEnsemble5 = ensemble5.predict(test)

# Evaluate accuracy

print(accuracy_score(test_labels, preds2))
print(accuracy_score(test_labels, preds3))
print(accuracy_score(test_labels, preds4))
print(accuracy_score(test_labels, preds5))
print(accuracy_score(test_labels, preds7))

print('Voting')
print(accuracy_score(test_labels, predVoting))

print('RandomForestClassifier')
print(accuracy_score(test_labels, predEnsemble1))
print('ExtraTreesClassifier')
print(accuracy_score(test_labels, predEnsemble2))
print('AdaBoostClassifier')
print(accuracy_score(test_labels, predEnsemble3))
print('GradientBoostingClassifier')
print(accuracy_score(test_labels, predEnsemble4))
print('GradientBoostingRegressor')
print(accuracy_score(test_labels, predEnsemble5))
