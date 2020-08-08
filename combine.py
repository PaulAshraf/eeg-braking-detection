import h5py
import math
import numpy as np
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.metrics
import sklearn.svm
import sklearn.decomposition
import sklearn.linear_model
import sklearn.neighbors
import sklearn.discriminant_analysis
import sklearn.tree
from csv import writer




emotiveNames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2','lead_brake','brake']
emotiveNumbers= [3,4,6,8,12,14,16,22,24,32,43,51,58,60,63,68]

window = 300

totalBreaks = 227
xName = 'P7'
yName = 'P8'

allVals = []
allVals1 = []
allVals2 = []
allVals3 = []
allVals4 = []

print('reading file 1')

with h5py.File('VPae.mat', 'r') as f:

    breakVals1 = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-2]:
        allVals1.append(f['cnt']['x'][dataSet][()])

print('reading file 2')

with h5py.File('VPbba.mat', 'r') as f:

    breakVals2 = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-2]:
        allVals2.append(f['cnt']['x'][dataSet][()])

print('reading file 3')

with h5py.File('VPgab.mat', 'r') as f:

    breakVals3 = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-2]:
        allVals3.append(f['cnt']['x'][dataSet][()])

print('reading file 4')

with h5py.File('VPgag.mat', 'r') as f:

    breakVals4 = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-2]:
        allVals4.append(f['cnt']['x'][dataSet][()])

print('finished reading file')

breakVals = np.concatenate((breakVals1, breakVals2, breakVals3, breakVals4))  #+ breakVals3

for i in range(len(allVals1)):
    allVals.append(np.concatenate((allVals1[i], allVals2[i], allVals3[i], allVals4[i])))

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

print('finished preping data')



gnb = sklearn.naive_bayes.GaussianNB()
clf1 = sklearn.svm.SVC()
clf2 = sklearn.svm.LinearSVC()
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
kneigh = sklearn.neighbors.KNeighborsClassifier()
dtc = sklearn.tree.DecisionTreeClassifier()
lr = sklearn.linear_model.LogisticRegression(max_iter=300)


pca = sklearn.decomposition.PCA()
pca.fit(features)
PCAfeatures = pca.transform(features)


train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(features, labels, test_size=0.2)

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
print('gnb: ',sklearn.metrics.accuracy_score(test_labels, preds1))
print('clf1: ',sklearn.metrics.accuracy_score(test_labels, preds2))
print('cl2: ',sklearn.metrics.accuracy_score(test_labels, preds3))
print('lda: ',sklearn.metrics.accuracy_score(test_labels, preds4))
print('kneigh: ',sklearn.metrics.accuracy_score(test_labels, preds5))
print('dtc: ',sklearn.metrics.accuracy_score(test_labels, preds6))
print('lr: ',sklearn.metrics.accuracy_score(test_labels, preds7))


