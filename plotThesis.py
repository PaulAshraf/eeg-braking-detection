import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from matplotlib import font_manager as fm, rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 13


# fpath = os.path.join(rcParams["datapath"], r"D:\\Program Installers\\laTeX font\\cm-unicode-0.7.0-ttf.tar\\cm-unicode-0.7.0-ttf\\cm-unicode-0.7.0\\cmunrm.ttf")
# prop = fm.FontProperties(fname=fpath)
# fname = os.path.split(fpath)[1]

data = np.genfromtxt('Thesis_ml_16_auc.csv', delimiter=',')
data2 = np.genfromtxt('Thesis_ml_16_auc_05.csv', delimiter=',')
data3 = np.genfromtxt('Thesis_ml_16_auc_01.csv', delimiter=',')




gnb = data[:,0].flatten()
svc = data[:,1].flatten()
linearSvc = data[:,2].flatten()
lda = data[:,3].flatten()
kneighbour = data[:,4].flatten()
dtc = data[:,5].flatten()
lr = data[:,6].flatten()

linearSvc2 = data2[:,2].flatten()
linearSvc3 = data3[:,2].flatten()

# voting = data[:,5].flatten()
# forest = data[:,6].flatten()
# extra = data[:,7].flatten()
# ada = data[:,8].flatten()


# maxVals = []
# for i in range(len(svc)):
#     maxVals.append(max([svc[i], linearSvc[i], lda[i], kneighbour[i], gnb[i], dtc[i], lr[i]]))



x = np.arange(100,1500 + 1,25)

N = 3

gnb = np.convolve(gnb, np.ones((N,))/N, mode='valid')
svc = np.convolve(svc, np.ones((N,))/N, mode='valid')
linearSvc = np.convolve(linearSvc, np.ones((N,))/N, mode='valid')
lda = np.convolve(lda, np.ones((N,))/N, mode='valid')
kneighbour = np.convolve(kneighbour, np.ones((N,))/N, mode='valid')
dtc = np.convolve(dtc, np.ones((N,))/N, mode='valid')
lr = np.convolve(lr, np.ones((N,))/N, mode='valid')

linearSvc2 = np.convolve(linearSvc2, np.ones((N,))/N, mode='valid')
linearSvc3 = np.convolve(linearSvc3, np.ones((N,))/N, mode='valid')


# voting = np.convolve(voting, np.ones((N,))/N, mode='valid')
# forest = np.convolve(forest, np.ones((N,))/N, mode='valid')
# extra = np.convolve(extra, np.ones((N,))/N, mode='valid')
# ada = np.convolve(ada, np.ones((N,))/N, mode='valid')

# maxVals = np.convolve(maxVals, np.ones((N,))/N, mode='valid')




plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc, label='0.1' ,c='red', alpha=1.0) 
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc2, label='0.05' ,c='blue', alpha=1.0) 
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc3, label='0.01' ,c='green', alpha=1.0) 

# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],svc,label='SVC',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc,label='Linear SVC',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lda,label='LDA',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],kneighbour,label='K Nearest Neighbour',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],dtc,label='DTC',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lr,label='LR',alpha=0.8)

# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],voting,label='Voting',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],forest,label='Random Forest Trees',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],extra,label='Extra Forest Trees',alpha=0.8)
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],ada,label='Ada Boost',alpha=0.8)


# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],maxVals,label='Maximum',c='black',ls='--')

edge = 0.9

plt.xlabel(r'Window Size (ms)')
plt.ylabel(r'AUC (Area Under the Curve) Score')
#plt.ylabel(r'Accuracy Score')


# plt.axhline(y=edge, color='black', linestyle='--')
# plt.axhline(y=edge+0.05, color='black', linestyle='-')


plt.ylim(0.5,1)
plt.xlim(100,1500)

plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.show()


