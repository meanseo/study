from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
# 실습!!!

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000,)

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

#print(x_train.shape)       #(60000, 784)
#print(x_test.shape)        #(10000, 784)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(y_train.shape)   결과확인

scaler = MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()

x_train = scaler.fit_transform(x_train)     # 이미 위에서 2차원 변환해놓아서 다시 reshape해줄필요가 없어졌다.
x_test = scaler.transform(x_test)



#2.모델링

input1 = Input(shape=(784,))
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
# model.add(Dense(15,activation="relu")) #
# model.add(Dense(8,activation="relu")) #
# model.add(Dense(5))
# model.add(Dense(10,activation="softmax"))

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

############################################################################
# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# # print(datetime)

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
# model_path = "".join([filepath, 'k32_cifar10_', datetime, '_', filename])
#                    # ./_ModelCheckPoint/k32_cifar10_1206_1656_2500-0.3724.hdf5
#############################################################################

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")
acc = '{accuracy:.4f}'
fn = "".join([krtime,acc])

es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras34_2_fashion_mnist{fn}_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=100,validation_split=0.1111111111, callbacks=[es])#,mcp

model.save(f"./_save/keras34_2_fashion_mnist{krtime}.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
결과정리
                Minmax
loss :      0.36644041538238525
accuracy :  0.8702999949455261
'''