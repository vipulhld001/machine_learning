import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
#from tensorflow.keras.callbacks import tensorboard
import pickle
import datetime

#Model Name for tensorboard and for my ease

log_dir = "D:\DeepLear\CATSVSDOG\PetImages\logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='D:\DeepLear\CATSVSDOG\PetImages\logs',histogram_freq=1)
#NAME = "BagvsGoogle-64by2-{}".format(int(time.time))
#Tensorboard = tensorBoard(log_dir='logs/{}'.format(NAME))
#Importing mY dataSet
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = keras.Sequential()
model.add(Conv2D(64,(3,3), input_shape = X.shape[1:]))# 3by 3 window
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape = X.shape[1:]))# 3by 3 window
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten(input_shape=(80,80)))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
model.fit(X, y, batch_size=32, validation_split=0.3, epochs=50, callbacks=[tensorboard_callback])
model.save('64by2and80by80CAT50Wala-CNN.model')
