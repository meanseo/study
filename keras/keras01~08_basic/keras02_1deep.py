# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과 
# 시퀀셜 모델 layer를 순차적으로 추가
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

#2. 모델구성 layer와 parameter를 추가하여 deep러닝으로 만들어본다.
model = Sequential()
model.add(Dense(5, input_dim=1))# input_dim=1 벡터의 갯수 곧 차원
model.add(Dense(4)) # 위에서 나온 출력이 그대로 넘어가니까 아래 줄에는 input 개수를 안써줘도 된다.
model.add(Dense(2))
model.add(Dense(1)) 
# 모델층을 두껍게해서 다중신경망을 형성하여 그 뒤 컴파일하고 예측을 해보면
# 단일신경망일때에 비하여 훈련량epochs를 훨씬 줄여도 loss값을 구할수있다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y) # 평가하다.
print('loss : ', loss)
result = model.predict([4]) # 새로운 x값을 predcit한 결과 
print('4의 예측값 : ', result)

'''
loss :  0.13166004419326782
4의 예측값 :  [[3.2348137]]
'''