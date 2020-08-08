import h5py
import math
from csv import writer
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


emotiveNames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2','lead_brake','gas','brake']
emotiveNumbers= [3,4,6,8,12,14,16,22,24,32,43,51,58,60,63,67,68]

# window = 300

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


AllResults = []

gnb = GaussianNB()
dtc = DecisionTreeClassifier()
###
clf1 = svm.SVC()
clf2 = svm.LinearSVC()
lda = LDA()
kneigh = KNeighborsClassifier()
lr = LogisticRegression(max_iter=300)

sc = StandardScaler()


for window in range(20,300 + 1, 5):

    print('RUN #',window)

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

    threshold = 0.1

    for i in range(len(breakVals)):

        if breakVals[i]>threshold and not isBraking:
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

    print('finished preping data for ',window)

    


    # pca = decomposition.PCA(n_components=i)
    # pca.fit(features)
    # pcafeatures = pca.transform(features)

    features = sc.fit_transform(features)

    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    # Train our classifier
    model1 = gnb.fit(train, train_labels)
    model2 = clf1.fit(train, train_labels)
    model3 = clf2.fit(train, train_labels)
    model4 = lda.fit(train, train_labels)
    model5 = kneigh.fit(train, train_labels)
    model6 = dtc.fit(train, train_labels)
    model7 = lr.fit(train, train_labels)


    # Make predictions
    preds1 = gnb.predict(test)
    preds2 = clf1.predict(test)
    preds3 = clf2.predict(test)
    preds4 = lda.predict(test)
    preds5 = kneigh.predict(test)
    preds6 = dtc.predict(test)
    preds7 = lr.predict(test)


    # Evaluate accuracy
    print(roc_auc_score(test_labels, preds1))
    print(roc_auc_score(test_labels, preds2))
    print(roc_auc_score(test_labels, preds3))
    print(roc_auc_score(test_labels, preds4))
    print(roc_auc_score(test_labels, preds5))
    print(roc_auc_score(test_labels, preds6))
    print(roc_auc_score(test_labels, preds7))

    AllResults.append([ roc_auc_score(test_labels, preds1), 
                        roc_auc_score(test_labels, preds2), 
                        roc_auc_score(test_labels, preds3),
                        roc_auc_score(test_labels, preds4),
                        roc_auc_score(test_labels, preds5),
                        roc_auc_score(test_labels, preds6), 
                        roc_auc_score(test_labels, preds7)])


with open("Thesis_ml_16_auc_01.csv","w+",newline='') as f:
    csvWriter = writer(f,delimiter=',')
    csvWriter.writerows(AllResults)

# 1 = VPae
# 2 = VPbad
# 3 = VPbba
# 4 = VPdx
# 5 = VPgab
# 6 = VPgag
# 7 = VPgam
# 8 = VPja