
import h5py
import math
import numpy as np
from csv import writer
import statistics


emotiveNames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2','lead_brake','gas','brake']
emotiveNumbers= [3,4,6,8,12,14,16,22,24,32,43,51,58,60,63,67,68]


allVals = []

print('reading file')

with h5py.File('VPae.mat', 'r') as f:

    breakVals = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]
    
    for dataSet in emotiveNumbers[0:len(emotiveNumbers)-3]:
        allVals.append(f['cnt']['x'][dataSet][()])

    allVals.append(f['cnt']['x'][emotiveNumbers[emotiveNames.index('gas')]][()])
    allVals.append(f['cnt']['x'][emotiveNumbers[emotiveNames.index('brake')]][()])


print('finished reading file')

time = []
lastBreak = 0

isBraking = False
finshed = False
brakingCounter = 0
barkingNum = 0

threshold = 0.1

for i in range(len(breakVals)):

    if breakVals[i]>threshold and not isBraking:
        lastBreak = i
        isBraking = True
        finshed = False
    
    if brakingCounter==600:
        isBraking = False
        brakingCounter = 0
        barkingNum+=1
    
    if isBraking:
        brakingCounter+=1 
        if allVals[-1][i]>threshold and not finshed:
            time.append((i - lastBreak)/200)
            finshed = True

print('finished preping data')

print(len(time))
print(statistics.mean(time))

with open("timeSaved.csv","w+",newline='') as f:
    csvWriter = writer(f,delimiter=',')
    csvWriter.writerows([[t] for t in time])


