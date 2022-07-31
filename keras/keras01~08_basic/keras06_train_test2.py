from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])             
y = np.array([1,2,3,4,5,6,7,8,9,10])

# train과 test비율을 8:2으로 분리하시오.
x_train =  x[:8]
x_test =  x[-2:]
y_train = y[:8]
y_test = y[-2:]

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

#2. 모델링하기
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss) # 결과값에서 나온 loss?
result = model.predict([11])
print('[11]의 예측값 : ', result)

'''
loss:  0.0007273558294400573
[11]의 예측값 :  [[10.962662]]
'''