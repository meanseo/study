# 함수형으로 바꿔보기

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.keras.layers import Dense, Input
import time

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)    # train과 test로 나누고나서 스케일링한다.

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)

'''
model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(30))
model.add(Dense(15,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
'''

input = Input(shape=(13, ))
dense1 = Dense(50)(input)
dense2 = Dense(30)(dense1)
dense3 = Dense(15, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
dense5 = Dense(5)(dense4)
output = Dense(1)(dense5)
model = Model(inputs=input, outputs=output)


model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.111111, callbacks=[es])

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")
print(krtime)

model.save(f"./_save/keras25_1_save_boston{krtime}.h5")
# model = load_model("./_save/keras25_1_save_boston.h5")

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)