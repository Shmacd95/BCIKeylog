import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

X = np.load('eegDat.npy')
y = np.load('eegMrk.npy')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#X_val, X_test, y_val, y_test = train_test_split(X, y, test_size = 0.50)

model = models.Sequential()
model.add(layers.Conv1D(32, 2, activation='elu', input_shape=X_train[0].shape))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(64, (2), activation='elu'))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(64, (2), activation='elu'))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()


model.compile(optimizer = 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

print(X_test.shape)
print(y_test.shape)

prediction = model.predict(X_test)

print(model.evaluate(X_test,y_test))

from sklearn.metrics import precision_recall_fscore_support

predictions = []
for i in range(0,len(prediction)):
	if prediction[i][0] > prediction[i][1]:
		predictions.append(0)
	else:
		predictions.append(1)

print(precision_recall_fscore_support(y_test,np.rint(predictions),average='macro'))
print(precision_recall_fscore_support(y_test,np.rint(predictions),average='micro'))
print(precision_recall_fscore_support(y_test,np.rint(predictions),average='weighted'))

