import numpy as np
import pandas as pd

colNames = ['0','1','2','3','4','5']
data1 = pd.read_csv('shane1_trimmed.csv', delimiter = ',', names=colNames)

data = data1.values

eegData = np.array([0,1,2,3,4])

for i in range(0, len(data)):
	if data[i][1] == " /marker" and data[i][2] <= 9 and data[i][2] >= 0:
		eegData[-1][-1] = data[i][2]
	elif data[i][1] != " /marker":
		eegData = np.vstack((eegData, np.hstack((data[i][2:], -1))))
		print(str(i))

print(eegData[1:].shape)
dataOut = pd.DataFrame(data=eegData[1:,:])
dataOut.to_csv('shane1_eeg.csv', sep=',')
