import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras import regularizers
from sklearn.preprocessing import StandardScaler


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
for name in emotiveNames[0:len(emotiveNumbers)-2]:
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
            single = []
            for k in range(window):
                single.append(allVals[j][i+k])
            feature.append(single)
        
        features.append(feature)
        labels.append(0)
    
    if brakingCounter==600:
        isBraking = False
        brakingCounter = 0
       
        feature = []
        for j in range(len(allVals)):
            single = []
            for k in range(window):
                single.append(allVals[j][i+k])
            feature.append(single)
        
        features.append(feature)
        labels.append(1)
        
        barkingNum+=1
    
    if isBraking:
        brakingCounter+=1

print('finished preping data')

features = np.array(features)
labels = np.array(labels)

samples, channels, windowData = features.shape
featuresFlat = features.reshape((samples,channels*windowData))
sc = StandardScaler()
featuresFlat = sc.fit_transform(featuresFlat)
features = featuresFlat.reshape((samples,channels,windowData,1))


train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# train = np.array(train)
# train_labels = np.array(train_labels)
# test = np.array(test)
# test_labels = np.array(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(channels, windowData, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, train_labels, epochs=1000, batch_size=16)

test_loss, test_acc = model.evaluate(test, test_labels)
print('Test accuracy:', test_acc)

