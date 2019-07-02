import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

device_name = "/gpu:0"
with tf.device(device_name):
	X = np.load("eegDat.npy")
	y = np.load("eegMrk.npy")
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(X_train[0].shape)),
		#keras.layers.Dense(256, activation=tf.nn.relu),
		keras.layers.Dense(128, activation=tf.nn.relu),
		#keras.layers.Dense(64, activation=tf.nn.relu),
		#keras.layers.Dense(32, activation=tf.nn.relu),
		#keras.layers.Dense(16, activation=tf.nn.relu),
		#keras.layers.Dense(8, activation=tf.nn.relu),
		#keras.layers.Dense(4, activation=tf.nn.relu),
		#keras.layers.Dense(2, activation=tf.nn.relu),
		keras.layers.Dense(1, activation=tf.nn.softmax)
	])
	optimizer = tf.keras.optimizers.RMSprop(0.001)

	#model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])


	model.summary()

	model.fit(X_train, y_train, epochs=1000)

	tmp, test_acc = model.evaluate(X_test, y_test)
	print('Test accuracy:', test_acc)
