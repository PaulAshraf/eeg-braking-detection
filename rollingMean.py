import h5py
import math
import matplotlib.pyplot as plt
# import pandas as pd
import csv
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




emotiveNames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2','lead_brake','brake']
emotiveNumbers= [3,4,6,8,12,14,16,22,24,32,43,51,58,60,63,68]

window = 300

totalBreaks = 227
xName = 'P7'
yName = 'P8'

allVals = []

print('Start Reading')

with h5py.File('VPae.mat', 'r') as f:

    breakVals = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-2]:
        allVals.append(f['cnt']['x'][dataSet][()])


print('Finshed Reading')


label_names = ['Braking','Normal']

feature_names = []
for name in emotiveNames[0:len(emotiveNumbers)-2]:
    for i in range(window):
        feature_names.append(name + str(i))

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

print('Finshed Preping')


AllResults = []

# pca = decomposition.PCA()
# pca.fit(features)
# features = pca.transform(features)

clf1 = svm.SVC()
clf2 = svm.LinearSVC()
lda = LDA()
kneigh = KNeighborsClassifier()
lr = LogisticRegression(max_iter=300)

for i in range(3,455,2):

    print('Running with N = ', i)

    for feature in features:
        N = i
        feature = np.convolve(feature, np.ones((N,))/N, mode='valid')


    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    

    # Train our classifier
    model2 = clf1.fit(train, train_labels)
    model3 = clf2.fit(train, train_labels)
    model4 = lda.fit(train, train_labels)
    model5 = kneigh.fit(train, train_labels)
    model7 = lr.fit(train, train_labels)


    # Make predictions
    preds2 = clf1.predict(test)
    preds3 = clf2.predict(test)
    preds4 = lda.predict(test)
    preds5 = kneigh.predict(test)
    preds7 = lr.predict(test)




    # Evaluate accuracy
    print('svc: ',accuracy_score(test_labels, preds2))
    print('lineaer SVC: ',accuracy_score(test_labels, preds3))
    print('lda: ',accuracy_score(test_labels, preds4))
    print('kneigh: ',accuracy_score(test_labels, preds5))
    print('lr: ',accuracy_score(test_labels, preds7))

    AllResults.append([ accuracy_score(test_labels, preds2), 
                        accuracy_score(test_labels, preds3),
                        accuracy_score(test_labels, preds4),
                        accuracy_score(test_labels, preds5),
                        accuracy_score(test_labels, preds7)])


with open("RollingMeanResults.csv","w+") as f:
    csvWriter = csv.writer(f,delimiter=',')
    csvWriter.writerows(AllResults)