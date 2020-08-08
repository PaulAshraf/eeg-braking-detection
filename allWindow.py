import h5py
import math
import matplotlib.pyplot as plt
from csv import writer
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score
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


emotiveNames = ['EOGv',
'Fp1',
'Fp2',
'AF3',
'AF4',
'EOGh',
'F7',
'F5',
'F3',
'F1',
'Fz',
'F2',
'F4',
'F6',
'F8',
'FT7',
'FC5',
'FC3',
'FC1',
'FCz',
'FC2',
'FC4',
'FC6',
'FT8',
'T7',
'C5',
'C3',
'C1',
'Cz',
'C2',
'C4',
'C6',
'T8',
'TP7',
'CP5',
'CP3',
'CP1',
'CPz',
'CP2',
'CP4',
'CP6',
'TP8',
'P9',
'P7',
'P5',
'P3',
'P1',
'Pz',
'P2',
'P4',
'P6',
'P8',
'P10',
'PO7',
'PO3',
'POz',
'PO4',
'PO8',
'O1',
'Oz',
'O2',
'EMGf',
'lead_gas',
'lead_brake',
'dist_to_lead',
'wheel_X',
'wheel_Y',
'gas',
'brake']


allVals = []

print('reading file')

with h5py.File('VPae.mat', 'r') as f:

    breakVals = f['cnt']['x'][emotiveNames.index('lead_brake')][()]
    
    for i in range(62):
        allVals.append(f['cnt']['x'][i][()])

    for i in range(64,69):
        allVals.append(f['cnt']['x'][i][()])

AllResults = []
gnb = GaussianNB()
dtc = DecisionTreeClassifier()
clf1 = svm.SVC(probability=True)
clf2 = svm.LinearSVC()
lda = LDA(solver='lsqr', shrinkage='auto')
kneigh = KNeighborsClassifier(weights='distance')
lr = LogisticRegression(max_iter=1000)

ensemble1 = RandomForestClassifier(n_estimators=1000, bootstrap = True, max_features = 'sqrt')
ensemble2 = ExtraTreesClassifier(n_estimators=1000, bootstrap = True, max_features = 'sqrt')
ensemble3 = AdaBoostClassifier(n_estimators=1000)

sc = StandardScaler()


print('finished reading file')


label_names = ['Braking','Normal']


for window in range(20,200 + 1, 5):

    print('RUN #',window)

    feature_names = []
    for name in emotiveNames[0:62]:
        for i in range(window):
            feature_names.append(name + str(i))

    for name in emotiveNames[64:69]:
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

    print('finished preping data for ',window)

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

    modelEnsemble1 = ensemble1.fit(train, train_labels)
    modelEnsemble2 = ensemble2.fit(train, train_labels)
    modelEnsemble3 = ensemble3.fit(train, train_labels)

    print('finished fitting data for ',window)


    # Make predictions
    preds1 = gnb.predict(test)
    preds2 = clf1.predict(test)
    preds3 = clf2.predict(test)
    preds4 = lda.predict(test)
    preds5 = kneigh.predict(test)
    preds6 = dtc.predict(test)
    preds7 = lr.predict(test)

    predEnsemble1 = ensemble1.predict(test)
    predEnsemble2 = ensemble2.predict(test)
    predEnsemble3 = ensemble3.predict(test)


    AllResults.append([ accuracy_score(test_labels, preds1), 
                        accuracy_score(test_labels, preds2), 
                        accuracy_score(test_labels, preds3),
                        accuracy_score(test_labels, preds4),
                        accuracy_score(test_labels, preds5),
                        accuracy_score(test_labels, preds6),
                        accuracy_score(test_labels, preds7),
                        accuracy_score(test_labels, predEnsemble1),
                        accuracy_score(test_labels, predEnsemble2),
                        accuracy_score(test_labels, predEnsemble3),
                        roc_auc_score(test_labels, preds1), 
                        roc_auc_score(test_labels, preds2), 
                        roc_auc_score(test_labels, preds3),
                        roc_auc_score(test_labels, preds4),
                        roc_auc_score(test_labels, preds5),
                        roc_auc_score(test_labels, preds6),
                        roc_auc_score(test_labels, preds7),
                        roc_auc_score(test_labels, predEnsemble1),
                        roc_auc_score(test_labels, predEnsemble2),
                        roc_auc_score(test_labels, predEnsemble3)
                         ])


with open("Thesis_64_all.csv","w+",newline='') as f:
    csvWriter = writer(f,delimiter=',')
    csvWriter.writerows(AllResults)

