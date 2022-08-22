from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
# 실습!!!

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

#print(x_train.shape)       #(50000, 3072)
#print(x_test.shape)        #(10000, 3072)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(y_train.shape)   결과확인

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()

x_train = scaler.fit_transform(x_train)     # 이미 위에서 2차원 변환해놓아서 다시 reshape해줄필요가 없어졌다.
x_test = scaler.transform(x_test)



#2.모델링

input1 = Input(shape=(x_train.shape[13],)) # 
dense1 = Dense(100)(input1)
dense6 = Dropout(0.2)(dense1)
dense2 = Dense(80)(dense6)
dense3 = Dense(60,activation="relu")(dense2)
dense7 = Dropout(0.4)(dense3)
dense4 = Dense(40,activation="relu")(dense7)
dense5 = Dense(20)(dense4)
output1 = Dense(10,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)


# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(30))
# model.add(Dropout(0.2))   
# model.add(Dense(15,activation="relu"))
# model.add(Dense(8,activation="relu"))
# model.add(Dense(5))
# model.add(Dense(10,activation="softmax"))

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")
#acc = '{accuracy:.4f}'
#fn = "".join([krtime,acc])

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras34_1_mnist{fn}_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=100,validation_split=0.1111111111, callbacks=[es])#,mcp

model.save(f"./_save/keras34_3_cifar10{krtime}.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
결과정리
                Minmax
loss :      1.4599390029907227
accuracy :  0.47870001196861267
'''