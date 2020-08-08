import h5py
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1619936 : number of values
# 11527 : 1st brake

dataClasses = ['EOGv','Fp1','Fp2','AF3','AF4','EOGh','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','EMGf','lead_gas','lead_brake','dist_to_lead','wheel_X','wheel_Y','gas','brake']

show = 51

x = np.arange(0,300)
y = np.zeros(300)

with h5py.File('VPae.mat', 'r') as f:
    allVals = f['cnt']['x'][show][()]
    breakVals = f['cnt']['x'][63][()]

isBraking = False
brakingCounter = 0
barkingNum = 0

for i in range(len(breakVals)):

    if breakVals[i]>0.1 and not isBraking:
        isBraking = True
        barkingNum+=1
    
    if brakingCounter==300:
        isBraking = False
        brakingCounter = 0
        print(y[0:5])
    
    if isBraking:
        y[brakingCounter] = ((y[brakingCounter] * (barkingNum - 1)) + allVals[i]) / barkingNum
        #y[brakingCounter] = allVals[i]
        brakingCounter+=1

print(barkingNum)

N = 15
filter1 = np.convolve(y, np.ones((N,))/N, mode='valid')

# fig, ax = plt.subplots(2)

# ax[0].plot(x, y)
# ax[0].set_title('Normal Average')

# ax[1].plot(x[0:len(x)-N+1], filter1)
# ax[1].set_title('Filtered Average')

# for ax in fig.get_axes():
#     ax.label_outer()

fig, ax = plt.subplots()
fig.suptitle(dataClasses[show])

ax.plot(x, y, '-r', label='Normal Average')
ax.plot(x[math.floor(N/2):len(x)-math.floor(N/2)], filter1, '-b', label='Filtered Average')

plt.legend(loc="upper right")

plt.show()