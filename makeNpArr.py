import numpy as np
import pandas as pd

colNames = ['0','1','2','3','4']
data1 = pd.read_csv('eeg_all.csv', delimiter = ',', names=colNames)
print(data1.shape)
data = data1.values[1:]
for i in range(0,len(data)):
	if data[i][-1] == -1:
		data[i][-1] = 0
	else:
		data[i][-1] = 1
print(data.shape)
count = 0
breaks = []
for i in range(0,len(data)):
	if data[i][-1] == 1 and (count % 250 == 0 or count % 250 == 249):
		count += 1
		breaks.append(i)
	elif data[i][-1] == 1:
		count += 1

print(count)
print(breaks)

datOut = np.vstack((data[breaks[0] - 256:breaks[1] + 256],data[breaks[2] - 256:breaks[3] + 256],data[breaks[4] - 256:breaks[5] + 256],data[breaks[6] - 256:breaks[7] + 256]))

print(datOut.shape)

#np.save("eeg_all.npy", datOut)
