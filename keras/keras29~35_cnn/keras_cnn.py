from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', input_shape=(10,10,1)))
model.add(MaxPooling2D())
model.add(Conv)