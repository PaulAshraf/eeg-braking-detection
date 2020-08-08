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

print('reading file')

with h5py.File('VPae.mat', 'r') as f:

    breakVals = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-2]:
        allVals.append(f['cnt']['x'][dataSet][()])

print('finished reading file')

label_names = ['Braking','Normal']

feature_names = []
for name in emotiveNames[0:len(emotiveNumbers)-2]:
    for i in range(window):
        feature_names.append(name + str(i))



AllResults = []

gnb = sklearn.naive_bayes.GaussianNB()
clf1 = sklearn.svm.SVC()
clf2 = sklearn.svm.LinearSVC()
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
kneigh = sklearn.neighbors.KNeighborsClassifier()
dtc = sklearn.tree.DecisionTreeClassifier()
lr = sklearn.linear_model.LogisticRegression(max_iter=300)





for window in range(20, 300 + 1, 10):

    features = []
    labels = []

    isBraking = False
    brakingCounter = 0
    barkingNum = 0

    print('preping data for window of size = ',window)

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


    pca = sklearn.decomposition.PCA(n_components=209)
    pca.fit(features)
    PCAfeatures = pca.transform(features)

    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(PCAfeatures, labels, test_size=0.2)

    

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
    print('clf1: ',sklearn.metrics.accuracy_score(test_labels, preds2))
    print('cl2: ',sklearn.metrics.accuracy_score(test_labels, preds3))
    print('lda: ',sklearn.metrics.accuracy_score(test_labels, preds4))
    print('kneigh: ',sklearn.metrics.accuracy_score(test_labels, preds5))
    print('lr: ',sklearn.metrics.accuracy_score(test_labels, preds7))

    AllResults.append([ sklearn.metrics.accuracy_score(test_labels, preds2), 
                        sklearn.metrics.accuracy_score(test_labels, preds3),
                        sklearn.metrics.accuracy_score(test_labels, preds4),
                        sklearn.metrics.accuracy_score(test_labels, preds5),
                        sklearn.metrics.accuracy_score(test_labels, preds7)])


with open("WindowResultsPCA209.csv","w+") as f:
    csvWriter = writer(f,delimiter=',')
    csvWriter.writerows(AllResults)