# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3,5,4])
y =  np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

model.fit(x, y, epochs=2800, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x, y) # 평가하다.
print('loss : ', loss)
result = model.predict([5]) # 새로운 x값을 predcit한 결과 
print('5의 예측값 : ', result)

'''
loss :  0.38053059577941895
6의 예측값 :  [[5.7355747]]
'''