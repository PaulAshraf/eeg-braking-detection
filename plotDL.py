

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import font_manager as fm, rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 16



data1 = np.genfromtxt('bestCNN.csv', delimiter=',')
data2 = np.genfromtxt('bestDL10K.csv', delimiter=',')
data3 = np.genfromtxt('bestDL10K2.csv', delimiter=',')
data4 = np.genfromtxt('ANN_100_epoch.csv', delimiter=',')

cnnData = np.genfromtxt('bestCNN_true.csv', delimiter=',')

dl1 =  data1[:,1].flatten()
dl2 =  data2[:,1].flatten()
dl3 =  data3[:,1].flatten()
dl4 =  data4[:,1].flatten()

cnn =  cnnData[:,1].flatten()


# maxVals = []
# for i in range(len(svc)):
#     maxVals.append(max([svc[i], linearSvc[i], lda[i], kneighbour[i], lr[i], cnn[i]]))

edge = 0.9


x = np.arange(100,1000 + 1,25)

N = 3

dl1 = np.convolve(dl1, np.ones((N,))/N, mode='valid')
dl2 = np.convolve(dl2, np.ones((N,))/N, mode='valid')
dl3 = np.convolve(dl3, np.ones((N,))/N, mode='valid')
dl4 = np.convolve(dl4, np.ones((N,))/N, mode='valid')

cnn = np.convolve(cnn, np.ones((N,))/N, mode='valid')

# maxVals = np.convolve(maxVals, np.ones((N,))/N, mode='valid')



# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],dl4[0:35],color='green',label='100 epoch')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],dl1[0:35],color='red',label='1000 epoch')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],dl2[0:35],color='red',label='batch_size = 363')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],dl3[0:35],color='blue',label='batch_size = 16')

plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],cnn[0:35],color='red')


plt.xlabel('Window Size (ms)')
plt.ylabel('Accuracy Score')

plt.ylim(0.5,1)
plt.xlim(100,1000)

# plt.legend(fontsize='small')

plt.grid(b=True, which='major', color='#666666', linestyle='--')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.show()



