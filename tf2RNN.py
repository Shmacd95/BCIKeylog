import tensorflow as tf
import numpy as np
import math

datIn = np.load('eeg_all.npy')
dat_train = datIn[:math.ceil(len(datIn) * 0.8)]
dat_test = datIn[math.ceil(len(datIn) * 0.8):]

BUFFER_SIZE = 10000
BATCH_SIZE = 64

model = tf.keras.Sequential([
    tf.keras.layers.Activation(activation='relu', input_shape=[4]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(dat_train, epochs=10)

test_loss, test_acc = model.evaluate(dat_test)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
