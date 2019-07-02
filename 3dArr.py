import numpy as np
import pandas as pd

colNames = ['0','1','2','3','4']
data1 = pd.read_csv('eeg_all.csv', delimiter = ',', names=colNames)

from sklearn import preprocessing
data = data1.values[1:]
tmp = data[:,-1]
data = data[:,:-1]
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
#print(data)
eegDat = np.zeros((2000,340,4))
mrks = np.zeros((2000,1))
count = 0

for i in range(0,len(data)):
	if tmp[i] != -1:
		eegDat[count] = data[(i-170):(i+170)]
		mrks[count] = 1
		count += 1
		eegDat[count] = data[(i+170):(i+510)]
		mrks[count] = 0
		count += 1

print(data)
np.save("eegDat_morebefore.npy",eegDat)
