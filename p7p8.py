import h5py
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1619936 : number of values
# 11527 : 1st brake

dataClasses = ['EOGv','Fp1','Fp2','AF3','AF4','EOGh','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','EMGf','lead_gas','lead_brake','dist_to_lead','wheel_X','wheel_Y','gas','brake']
emotiveNames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2','lead_brake','brake']
emotiveNumbers= [3,4,6,8,12,14,16,22,24,32,43,51,58,60,63,68]

totalBreaks = 227
xName = 'P7'
yName = 'P8'


p7Braking = np.zeros(totalBreaks)
p8Braking = np.zeros(totalBreaks)

p7AvgBraking = [0]
p8AvgBraking = [0]

p7NonBraking = np.zeros(totalBreaks)
p8NonBraking = np.zeros(totalBreaks)

p7AvgNonBraking = [0]
p8AvgNonBraking = [0]

with h5py.File('VPae.mat', 'r') as f:
    p7Vals = f['cnt']['x'][emotiveNumbers[emotiveNames.index(xName)]][()]
    p8Vals = f['cnt']['x'][emotiveNumbers[emotiveNames.index(yName)]][()]
    breakVals = f['cnt']['x'][emotiveNumbers[emotiveNames.index('lead_brake')]][()]

isBraking = False
brakingCounter = 0
barkingNum = 0

for i in range(len(breakVals)):

    if breakVals[i]>0.1 and not isBraking:
        isBraking = True
        
        p7Braking[barkingNum] = sum(p7Vals[i+100:i+150+1])/50
        p8Braking[barkingNum] = sum(p8Vals[i+100:i+150+1])/50

        p7AvgBraking[0] = (p7AvgBraking[0] *  barkingNum + p7Vals[i+125]) / (barkingNum + 1)  
        p8AvgBraking[0] = (p8AvgBraking[0] *  barkingNum + p8Vals[i+125]) / (barkingNum + 1)  
    
    if brakingCounter==200:
        isBraking = False
        brakingCounter = 0
        p7NonBraking[barkingNum] = sum(p7Vals[i:i+50+1])/50
        p8NonBraking[barkingNum] = sum(p8Vals[i:i+50+1])/50

        p7AvgNonBraking[0] = (p7AvgNonBraking[0] *  barkingNum + p8Vals[i]) / (barkingNum + 1)  
        p8AvgNonBraking[0] = (p8AvgNonBraking[0] *  barkingNum + p8Vals[i]) / (barkingNum + 1)  

        barkingNum+=1
    
    if isBraking:
        brakingCounter+=1

print(barkingNum)

# fig, ax = plt.subplots(2, sharex=True, sharey=True)

# ax[0].scatter(p7Braking, p8Braking, s=4, c='red', label='Braking')
# ax[1].scatter(p7NonBraking, p8NonBraking, s=4, c='blue', label='Normal')

# for ax in fig.get_axes():
#     # ax.label_outer()
#     ax.legend(loc='upper left')

# # ax[0].set(xlabel=xName)
# # ax[1].set(xlabel=xName, ylabel=yName)

# plt.show()

fig, ax = plt.subplots()

ax.scatter(p7Braking, p8Braking, s=4, c='red', label='Braking', alpha=0.5)
ax.scatter(p7NonBraking, p8NonBraking, s=4, c='blue', label='Normal', alpha=0.5)
ax.scatter(p7AvgBraking,p8AvgBraking, s=100, c='black', label='Average Braking', marker='X')
ax.scatter(p7AvgNonBraking,p8AvgNonBraking, s=100, c='green', label='Average Normal', marker='X')

ax.legend(loc='upper left')

ax.set(xlabel=xName)
ax.set(xlabel=xName, ylabel=yName)

plt.show()