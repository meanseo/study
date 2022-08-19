# callbacks 함수의 EarlyStopping 기능을 이용하여 diabets 데이터 모델링해보기 
# 숙제 및 지금까지 배운내용 총 정리.

from tensorflow.keras.models import Sequential          # 신경망 모델링할 모델의 종류 Sequential
from tensorflow.keras.layers import Dense               # 모델링에서 사용할 레이어의 종류 Dense 
from tensorflow.keras.layers import Activation
from sklearn.datasets import load_diabetes              # 싸이킷런 라이브러리의 datasets클래스의 diabets함수 불러옴
from sklearn.model_selection import train_test_split    # 데이터를 train과 test로 0.0~1.0 사이의 비율로 분할 및 랜덤분류 기능
from sklearn.metrics import r2_score                    # y_predict값과 y_test값을 비교하여 점수매김. 0.0~1.0 및 - 값도 나옴.
import matplotlib.pyplot as plt                         # 데이터를 시각화 시켜주는 기능.
from tensorflow.keras.callbacks import EarlyStopping    # training 조기종료를 도와주는 기능 여러 옵션들이 있다.

dataset =  load_diabetes()
x =  dataset.data
y = dataset.target
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, shuffle=True, random_state=66)

model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Activation('relu'))
model.add(Dense(90))
model.add(Activation('relu'))
model.add(Dense(80))
model.add(Activation('relu'))
model.add(Dense(70))
model.add(Activation('relu'))
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=1,validation_split=0.25, callbacks=[es])

loss = model.evaluate(x_test,y_test)
print('평가 loss의 값 : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print('r2스코어는', r2)

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

#########################################################################################################################

#1. 데이터 로드 및 정제
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=66)

#2. 모델링  여러번해서 좋은 값 찾아야함.
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Activation('relu'))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Activation('softmax'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
# val_loss를 관측하고 50번안에 최저값이 갱신되지 않으면 훈련을 중단하고 가장 좋았을때의 w값을 복원
# 컴파일해보면 마지막에 Restoring model weights from the end of the best epoch. 라는 메시지를 출력
# baseline = None ,  모델이 달성해야하는 최소한의 기준값,정확도을 선정합니다. 정확도를 측정하는 기준은? 0.0~1.0

# True값 넣고 evaluate했을때  1.loss: 3447.219482421875  r2: 0.5668603667030361  2.loss: 3641.060302734375   r2: 0.5425044431962383   3.loss: 3614.39453125  r2: 0.5458550011751824
# False값 넣고 evalutae했을때 1.loss: 3701.3193359375    r2: 0.5349329860627352  2.loss: 3321.02294921875    r2: 0.5827168650569212   3.loss: 3441.029541015625   r2: 0.5676381480025825
# 큰 차이가 없는 이유. EarlyStopping은 최적의 weights값을 복원해서 저장한다. <-- 기록하고 저장해서 evaluate 할때 최적의 값으로 계산한다.
# 값을 저장하려면 ModelCheckpoint 함수를 써야한다.


hist = model.fit(x_train,y_train,epochs=10000, batch_size=1, validation_split=0.111111, callbacks=[es]) 
# loss와 똑같이 관측하기 위해 일단 저장.
#[ ]로 감싸주는 이유 : 2개이상 값을 넣어주려고 나중에 modelcheckpoint 등을 추가

# hist 안에 수 많은 정보들이 담긴다.
# print("-------------------------------------------")
# print(hist)   # 자료형이 나온다.
# print("-------------------------------------------")
# print(hist.history)  # loss 값과 var_loss값이 dic형태로 저장되어 있다. epoch 값만큼의 개수가 저장되어 있다 ->> 1epoch당 값을 하나씩 다 저장한다.
# print("-------------------------------------------")
# print(hist.history['loss']) # hist.history에서 loss키 값의 value들을 출력해준다.
# print("-------------------------------------------")
# print(hist.history['val_loss']) # hist.history var_loss키 값의 value들을 출력해준다.
# print("-------------------------------------------")


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
# 컴파일 단계에서 도출된 weight 및 bias값에 xy_test를 넣어서 평가만. 해보고 거기서 나온 loss들을 저장.


print('평가만 해본 후 나온 loss의 값 : ', loss)
# val_loss와 loss의 차이가 적을수록 validation data가 더 최적의 weights를 도출시켜줘서 실제로 평가해봐도 차이가 적게 나온다는 말이므로 차이가 적을수록 좋다.
#model.evaluate도  model.fit처럼 수많은 값들을 loss안에 담아주는 줄 알았다 근데 보려고했더니
#print(loss.history) loss도 hist처럼 history볼수 있을줄 알았는데 'float' object has no attribute 'history' 라고 나온다.  <---------------------------------------------------------------------------- 질문할거

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print('r2스코어는', r2)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()