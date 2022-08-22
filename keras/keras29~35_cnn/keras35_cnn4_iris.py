from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import inspect, os

#1 데이터 정제작업 !!
datasets = load_iris()
x = datasets.data           
y = datasets.target         

#print(x.shape) # x형태  (150, 4)   
#print(y.shape) # y형태  (150,)
#print(np.unique(y, return_counts=True))    # 0이 50개 1이 50개 2가 50개이다. 다중분류

# 이 데이터는 columns가 4개라서 굳이 손 안보고 그냥 씀.
#print(type(x)) #numpy형태라 변환 필요없음.


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()     

x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,1)
x_test = scaler.transform(x_test).reshape(len(x_test),2,2,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델링

model = Sequential()
model.add(Conv2D(4,kernel_size=(2,2),strides=1,padding='same', input_shape=(2,2,1), activation='relu'))  # 2,2,1                                                                      # 1,1,10
model.add(MaxPooling2D(2,2))                                                                            # 1 1 1                               
model.add(Flatten())       
model.add(Dense(60))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=5,validation_split=0.111111, callbacks=[es])#,mcp



#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

acc= str(round(loss[1], 4))

# a = inspect.getfile(inspect.currentframe())  #현재 파일이 위치한 경로 + 현재 파일 명
# print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
# print(a.split("\\")[-1]) #현재 파일 명

model.save(f"./_save/keras35_4_iris_acc_Min_{acc}.h5")


'''
결과정리
            Minmax                  standard
loss:       0.05541 
accuracy:     1.0
'''