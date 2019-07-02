import numpy as np
import pandas as pd

colNames = ['0','1','2','3','4']
data1 = pd.read_csv('shane1_eeg.csv', delimiter = ',', names=colNames)
data2 = pd.read_csv('shane2_eeg.csv', delimiter = ',', names=colNames)
data3 = pd.read_csv('shane3_eeg.csv', delimiter = ',', names=colNames)
data4 = pd.read_csv('shane4_eeg.csv', delimiter = ',', names=colNames)


data11 = data1.values
data22 = data2.values
data33 = data3.values
data44 = data4.values

dataOut = np.vstack((data11[1:],data22[1:],data33[1:],data44[1:]))

for i in range(0,len(data11)):
	if data11[i][-1] == 7:
		dataOut = data11[i-85:i+255]
		break


print(dataOut.shape)
print(dataOut)
dataOut = pd.DataFrame(data=dataOut[1:,:])
dataOut.to_csv('slice.csv', sep=',')




