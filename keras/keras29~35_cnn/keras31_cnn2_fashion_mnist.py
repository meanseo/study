from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import fashion_mnist # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import get_dummies


#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)
#print(x_test.shape)

#print(np.unique(y_train, return_counts=True))


y_train = get_dummies(y_train)
y_test = get_dummies(y_test)
#print(y_train.shape,y_test.shape)

#2. 모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras30_mnist_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.2, callbacks=[es])#,mcp

#model.save(f"./_save/keras30_save_mnist.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :      0.31123247742652893       0.3209112584590912      0.32448408007621765
# accuracy :  0.8934000134468079        0.8866000175476074      0.888700008392334