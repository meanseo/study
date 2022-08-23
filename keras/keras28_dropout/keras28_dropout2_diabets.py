from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.keras.layers import Dense, Input
import time

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)   

'''
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(80,activation='relu')) #
model.add(Dense(60))
model.add(Dense(40,activation='relu')) #
model.add(Dense(20))
model.add(Dense(1))
model.summary()
'''
input = Input(shape=(10,))
dense1 = Dense(100)(input)
dense2 = Dense(80, activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense2)
dense3 = Dense(60)(dropout1)
dense4 = Dense(40, activation='relu')(dense3)
dropout2 = Dropout(0.4)(dense4)
dense5 = Dense(20)(dense4)
output = Dense(1)(dense5)
model = Model(inputs=input, outputs=output)

model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.1111111, callbacks=[es])

# model.save("./_save/keras25_2_save_diabets.h5")
#model = load_model("./_save/keras25_2_save_diabets.h5")

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

# print("======================= 2. load_model 출력 ======================")
# model2 = load_model(f"./_save/keras26_2_save_diabets{krtime}.h5")
# loss2 = model2.evaluate(x_test,y_test)
# print('loss2 : ', loss2)

# y_predict2 = model2.predict(x_test)

# r2 = r2_score(y_test,y_predict2) 
# print('r2스코어 : ', r2)

# print("====================== 3. mcp 출력 ============================")
# model3 = load_model(f'./_ModelCheckPoint/keras26_2_diabets{krtime}_MCP.hdf5')
# loss3 = model3.evaluate(x_test,y_test)
# print('loss3 : ', loss3)

# y_predict3 = model3.predict(x_test)

# r2 = r2_score(y_test,y_predict3) 
# print('r2스코어 : ', r2)