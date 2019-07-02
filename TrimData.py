import numpy as np
import pandas as pd

colNames = ['0','1','2','3','4','5','6','7']
data1 = pd.read_csv('shane4.csv', delimiter = ',', names=colNames)

data = data1.values

mrk = data[10605,1]
eeg = data[1,1]

trimmedData = np.array([0,1,2,3,4,5,6,7])
for row in range(0,data.shape[0]):
    print(row)
    if (data[row,1] == eeg) or (data[row,1] == mrk):
        trimmedData = np.vstack((trimmedData, data[row,:]))

print(trimmedData.shape)

dataOut = pd.DataFrame(data=trimmedData[1:,1:6])
dataOut.to_csv('shane4_trimmed.csv', sep=',')
