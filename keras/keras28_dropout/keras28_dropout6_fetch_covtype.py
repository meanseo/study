from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from pandas import get_dummies
import time

#1.데이터 로드 및 정제

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)       #어떤 비율로 변환할지 계산해줌.
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링

input1 = Input(shape=(54,))
dense1 = Dense(100,activation="relu")(input1)
drop1  = Dropout(0.2)(dense1)
dense2 = Dense(80)(drop1)
dense3 = Dense(60,activation="relu")(dense2)
drop2  = Dropout(0.4)(dense3)
dense4 = Dense(40)(drop2)
dense5 = Dense(20)(dense4)
output1 = Dense(7,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor="val_loss", patience=10, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_6_fetch_covtype{krtime}_MCP.hdf5')

model.fit(x_train,y_train,epochs=10000, batch_size=100000,validation_split=0.11111111, callbacks=[es])#,mcp


#model.save(f"./_save/keras26_6_save_covtype{krtime}.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
