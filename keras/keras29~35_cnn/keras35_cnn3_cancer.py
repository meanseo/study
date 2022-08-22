from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#1 데이터 정제작업 !!
datasets = load_breast_cancer()
x = datasets.data           
y = datasets.target         

print(x.shape) # x형태  (569, 30)   
print(y.shape) # y형태  (569,)
#print(np.unique(y, return_counts=True))    # 0이 212개 1이 357개이다. 이중분류

# 이 데이터는 columns가 30개라서 굳이 손 안보고 그냥 씀.
#print(type(x)) #numpy형태라 변환 필요없음.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()     

x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,6,1)
x_test = scaler.transform(x_test).reshape(len(x_test),5,6,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델링

model = Sequential()
model.add(Conv2D(4,kernel_size=(2,2),strides=1,padding='same', input_shape=(5,6,1), activation='relu'))    # 5,6,1 -> 5,6,4  사이즈 유지                                                                    # 1,1,10
model.add(Conv2D(4,kernel_size=(2,3), strides=1, padding='valid', activation='relu'))                       # 5,6,4 -> 4.4.4
model.add(MaxPooling2D(2,2))                                                                                # 4,4,4 -> 2.2.4
model.add(Conv2D(4,kernel_size=(2,2), strides=1, padding='valid', activation='relu'))                      # 2,2,4 -> 1,1,4
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=5,validation_split=0.111111, callbacks=[es])#,mcp



#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

acc= str(round(loss[1], 4))

model.save(f"./_save/keras35_3_cancer_acc_Min_{acc}.h5")


'''
결과정리
            Minmax                  standard
loss:     0.18027029931545258     0.4145790934562683
accuracy: 0.9473684430122375      0.8771929740905762
'''