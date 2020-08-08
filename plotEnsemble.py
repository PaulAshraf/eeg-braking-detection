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


data = np.genfromtxt('Book1.csv', delimiter=',')
dataCNN = np.genfromtxt('bestCNN.csv', delimiter=',')


svc = data[:,0].flatten()
linearSvc = data[:,1].flatten()
lda = data[:,2].flatten()
kneighbour = data[:,3].flatten()
lr = data[:,4].flatten()

voting = data[:,5].flatten()
forest = data[:,6].flatten()
extra = data[:,7].flatten()
ada = data[:,8].flatten()

cnn =  dataCNN[:,1].flatten()[0:37]

maxVals = []
for i in range(len(svc)):
    maxVals.append(max([svc[i], linearSvc[i], lda[i], kneighbour[i], voting[i], forest[i], extra[i], ada[i], lr[i], cnn[i]]))

edge = 0.9


x = np.arange(100,1000 + 1,25)

N = 3

svc = np.convolve(svc, np.ones((N,))/N, mode='valid')
linearSvc = np.convolve(linearSvc, np.ones((N,))/N, mode='valid')
lda = np.convolve(lda, np.ones((N,))/N, mode='valid')
kneighbour = np.convolve(kneighbour, np.ones((N,))/N, mode='valid')
lr = np.convolve(lr, np.ones((N,))/N, mode='valid')

cnn = np.convolve(cnn, np.ones((N,))/N, mode='valid')

voting = np.convolve(voting, np.ones((N,))/N, mode='valid')
forest = np.convolve(forest, np.ones((N,))/N, mode='valid')
extra = np.convolve(extra, np.ones((N,))/N, mode='valid')
ada = np.convolve(ada, np.ones((N,))/N, mode='valid')

maxVals = np.convolve(maxVals, np.ones((N,))/N, mode='valid')






plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],svc,label='SVC',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],linearSvc,label='Linear SVC',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lda,label='LDA',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],kneighbour,label='K Nearest Neighbour',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],lr,label='LR',alpha=0.5)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],cnn,label='ANN',alpha=0.5)

plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],voting,label='Voting',alpha=0.8)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],forest,label='Random Forest Trees',alpha=0.8)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],extra,label='Extra Forest Trees',alpha=0.8)
plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],ada,label='Ada Boost',alpha=0.8)


plt.plot(x[math.floor(N/2):len(x)-math.floor(N/2)],maxVals,label='Maximum',c='black',ls='--')


plt.xlabel('Window Size (ms)')
plt.ylabel('Accuracy Score')

plt.axhline(y=edge, color='r', linestyle='-')

plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.show()


