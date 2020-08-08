# import math
# import numpy as np
# import matplotlib.pyplot as plt


# data = np.genfromtxt('PCAResults.csv', delimiter=',')


# svc = data[:,1].flatten()
# linearSvc = data[:,2].flatten()
# lda = data[:,4].flatten()
# kneighbour = data[:,5].flatten() 
# lr = data[:,7].flatten()

# x = np.arange(1,454+1)

# N = 21

# svc = np.convolve(svc, np.ones((N,))/N, mode='valid')
# linearSvc = np.convolve(linearSvc, np.ones((N,))/N, mode='valid')
# lda = np.convolve(lda, np.ones((N,))/N, mode='valid')
# kneighbour = np.convolve(kneighbour, np.ones((N,))/N, mode='valid')
# lr = np.convolve(lr, np.ones((N,))/N, mode='valid')

# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],svc,label='SVC')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc,label='Linear SVC')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lda,label='LDA')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],kneighbour,label='K Nearest Neighbour')
# plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lr,label='LR')

# plt.xlabel('PCA n_components')
# plt.ylabel('Accuracy Score')

# plt.legend()
# plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt

svc = []
linearSvc = []
lda = []
kneighbour = []
lr = []

for i in range(1,9):
    subject = np.genfromtxt('bestWindowScaler'+str(i)+'.csv', delimiter=',')
    svc.append(subject[:,0].flatten())
    linearSvc.append(subject[:,1].flatten())
    lda.append(subject[:,2].flatten())
    kneighbour.append(subject[:,3].flatten())
    lr.append(subject[:,4].flatten())


dataCNN = np.genfromtxt('bestCNN.csv', delimiter=',')

# svc = np.array(svc)
# linearSvc = np.array(svc)
# lda = np.array(svc)
# kneighbour = np.array(svc)
# lr = np.array(svc)


svc = np.average(svc, axis=0)
linearSvc = np.average(linearSvc, axis=0)
lda = np.average(lda, axis=0)
kneighbour = np.average(kneighbour, axis=0)
lr = np.average(lr, axis=0)

print(svc)
print(lr)

cnn =  dataCNN[:,1].flatten()

maxVals = []
for i in range(len(svc)):
    maxVals.append(max([svc[i], linearSvc[i], lda[i], kneighbour[i], lr[i], cnn[i]]))

edge = 0.9


x = np.arange(100,1500 + 1,25)

N = 1

# svc = np.convolve(svc, np.ones((N,))/N, mode='valid')
# linearSvc = np.convolve(linearSvc, np.ones((N,))/N, mode='valid')
# lda = np.convolve(lda, np.ones((N,))/N, mode='valid')
# kneighbour = np.convolve(kneighbour, np.ones((N,))/N, mode='valid')
# lr = np.convolve(lr, np.ones((N,))/N, mode='valid')

# cnn = np.convolve(cnn, np.ones((N,))/N, mode='valid')

# maxVals = np.convolve(maxVals, np.ones((N,))/N, mode='valid')


plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],svc,label='SVC',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc,label='Linear SVC',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lda,label='LDA',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],kneighbour,label='K Nearest Neighbour',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lr,label='LR',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],cnn,label='CNN',alpha=0.5)

plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],maxVals,label='Maximum',c='black',ls='--')


plt.xlabel('Window Size (ms)')
plt.ylabel('Accuracy Score')

plt.axhline(y=edge, color='r', linestyle='-')

plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.show()


