from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
import numpy as np
from tensorflow.keras.datasets import mnist # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from pandas import get_dummies
from sklearn.preprocessing import OneHotEncoder
# 평가지표 acc 
# 0.98 

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1) # reshape의 개념은 다 곱해서 일렬로 만든 후 다시 나눠서 재배열하는 개념
# CNN에서 계층 사이를 흐르는 데이터는 4차원
x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape)

# print(np.unique(y_train, return_counts=True))   # np.unique(y_train, return_counts=True) 하면 pandas의 value.counts와 같은 기능

enco = OneHotEncoder(sparse=False)

y_train = enco.fit_transform(y_train.reshape(-1,1)) 
# y_train = get_dummies(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_train[:5])
print(y_test[:5])

#2. 모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

'''
#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 
es = EarlyStopping(monitor="val_loss", patience=10, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras30_mnist_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.2, callbacks=[es])#,mcp
#model.save(f"./_save/keras30_save_mnist.h5")
#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
# loss :  0.058354027569293976          0.04385140538215637
# accuracy :  0.9817000031471252        0.9861000180244446
'''