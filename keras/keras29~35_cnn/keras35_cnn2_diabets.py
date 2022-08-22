from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 데이터 정제작업 !!
datasets = load_diabetes()
x = datasets.data           
y = datasets.target         

#print(x.shape) # x형태  (442, 10)    
#print(y.shape) # y형태  (442,)
#print(np.unique(y, return_counts=True))     # 종류가 셀수없이 많다. 회귀모델

# 이 데이터는 columns가 10개라서 굳이 손 안보고 그냥 씀.
#print(type(x)) numpy형태라 변환 필요없음.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)

scaler =StandardScaler()   #MinMaxScaler()RobustScaler()MaxAbsScaler()     

x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,5,1)
x_test = scaler.transform(x_test).reshape(len(x_test),2,5,1)


#2.모델링

model = Sequential()
model.add(Conv2D(4,kernel_size=(2,2),strides=1,padding='same', input_shape=(2,5,1), activation='relu'))    # 2,5,4                                                                        # 1,1,10
model.add(Conv2D(4,kernel_size=(1,2), strides=1, padding='valid', activation='relu'))                       # 2,4,4 
model.add(MaxPooling2D(2,2))                                                                                # 1,2,4
model.add(Conv2D(4,kernel_size=(1,2), strides=1, padding='valid', activation='relu'))                      # 1,1,4
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=5,validation_split=0.111111, callbacks=[es])#,mcp



#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

r2s = round(r2,4)

model.save(f"./_save/keras35_2_diabetes_r2_sta_{r2s}.h5")

'''
결과정리
            Minmax                  standard
loss:      2037.5010986328125        1935.361083984375
r2스코어:   0.5988315165767664       0.6189420650543955
'''