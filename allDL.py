import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler


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

window = 100

allVals = []

print('reading file')

with h5py.File('VPae.mat', 'r') as f:

    breakVals = f['cnt']['x'][emotiveNames.index('lead_brake')][()]
    
    for i in range(62):
        allVals.append(f['cnt']['x'][i][()])

    for i in range(64,69):
        allVals.append(f['cnt']['x'][i][()])

    


print('finished reading file')


label_names = ['Braking','Normal']

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

print('finished preping data')

sc = StandardScaler()
features = sc.fit_transform(features)


train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
train = np.array(train)
train_labels = np.array(train_labels)
test = np.array(test)
test_labels = np.array(test_labels)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(window * 67,)),
    keras.layers.Dense(500, activation=tf.nn.relu),
	keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, train_labels, epochs=100, batch_size=16)

test_loss, test_acc = model.evaluate(test, test_labels)
print('Test accuracy:', test_acc)
