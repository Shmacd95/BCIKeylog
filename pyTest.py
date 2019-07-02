import numpy as np
import pandas as pd

colNames = ['0','1','2','3','4']
data1 = pd.read_csv('eeg_all.csv', delimiter = ',', names=colNames)

data = data1.values
times = []

for i in range(0, len(data)):
	if data[i][-1] != -1:
		times.append(i)

print(times)
