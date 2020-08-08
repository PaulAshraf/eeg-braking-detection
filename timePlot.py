import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from matplotlib import font_manager as fm, rcParams
import statistics

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 13


data = np.genfromtxt('Thesis_ml_16_auc.csv', delimiter=',')

gnb = data[:,0].flatten()
svc = data[:,1].flatten()
linearSvc = data[:,2].flatten()
lda = data[:,3].flatten()
kneighbour = data[:,4].flatten()
dtc = data[:,5].flatten()
lr = data[:,6].flatten()

data2 = np.genfromtxt('timeSaved.csv', delimiter=',')

print(data2.shape)

timeArr = data2[:,].flatten()
time = statistics.mean(timeArr)



x = np.arange(0.5,0.95 + 0.001,0.01)

N = 1

gnb = np.convolve(gnb, np.ones((N,))/N, mode='valid')
svc = np.convolve(svc, np.ones((N,))/N, mode='valid')
linearSvc = np.convolve(linearSvc, np.ones((N,))/N, mode='valid')
lda = np.convolve(lda, np.ones((N,))/N, mode='valid')
kneighbour = np.convolve(kneighbour, np.ones((N,))/N, mode='valid')
dtc = np.convolve(dtc, np.ones((N,))/N, mode='valid')
lr = np.convolve(lr, np.ones((N,))/N, mode='valid')

timeSaved = [[] for _ in range(7)]


for i in x:
    found = [False for _ in range(7)]
    for j in range(len(gnb)):
        if gnb[j] > i and not found[0]:
            timeSaved[0].append((time * 1000) - (j*25 + 100))
            found[0] = True
        if svc[j] > i and not found[1]:
            timeSaved[1].append((time * 1000) - (j*25 + 100))
            found[1] = True
        if linearSvc[j] > i and not found[2]:
            timeSaved[2].append((time * 1000) - (j*25 + 100))
            found[2] = True
        if lda[j] > i and not found[3]:
            timeSaved[3].append((time * 1000) - (j*25 + 100))
            found[3] = True
        if kneighbour[j] > i and not found[4]:
            timeSaved[4].append((time * 1000) - (j*25 + 100))
            found[4] = True
        if dtc[j] > i and not found[5]:
            timeSaved[5].append((time * 1000) - (j*25 + 100))
            found[5] = True
        if lr[j] > i and not found[6]:
            timeSaved[6].append((time * 1000) - (j*25 + 100))
            found[6] = True
    for f in range(len(found)):
        if not found[f]:
            print(i,' ',f)
            timeSaved[f].append((time * 1000) - (1500))

print(len(x))
print(len(timeSaved[2]))


plt.plot([e*100 for e in x],timeSaved[0], label='GNB',  alpha=1.0) 
plt.plot([e*100 for e in x],timeSaved[1], label='SVC',  alpha=1.0) 
plt.plot([e*100 for e in x],timeSaved[2], label='LSVC', alpha=1.0) 
plt.plot([e*100 for e in x],timeSaved[3], label='LDA', alpha=1.0) 
plt.plot([e*100 for e in x],timeSaved[4], label='KNN', alpha=1.0) 
plt.plot([e*100 for e in x],timeSaved[5], label='DTC', alpha=1.0) 
plt.plot([e*100 for e in x],timeSaved[6], label='LR', alpha=1.0) 




plt.ylabel(r'Time Saved (ms)')

plt.xlabel(r'AUC Score')
#plt.ylabel(r'Accuracy Score')


plt.axhline(y=0, color='black', linestyle='-')


plt.xlim(50,100)
plt.ylim(-100,700)



plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)


plt.show()


