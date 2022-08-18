# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과 
# 시퀀셜 모델 layer를 순차적으로 추가
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # Dense 레이어 추가
# 앞의 숫자가 출력 노드 갯수 / input_dim=1 -> 입력 데이터의 차원이 1차원?? 아니면 입력 노드

#3. 컴파일, 훈련
# 컴퓨터가 알아듣게 훈련시키는 것,  그게 컴파일 y = wx + b 최적의 weight값을 빼기 위한 최소의 loss 값을 찾는다.
# mse(mean squared error): 회귀 용도의 모델을 훈련시킬 때 사용되는 손실함수
model.compile(loss='mse', optimizer='adam') # 평균 제곱 에러 mse 이 값은 작을수록 좋다. optimizer='adam'은 mse값(loss) 감축시키는 역할. 85점 이상이면 쓸만하다.

model.fit(x, y, epochs=4200, batch_size=1) # epochs 훈련 횟수을 의미, batch_size 몇 개씩 데이터를 넣을지 지정해줌. batch가 작을수록 값이 정교해짐
#위의 데이터들로 훈련하겠다 fit. 

#4. 평가, 예측
loss = model.evaluate(x, y) # 평가하다.
print('loss : ', loss)
result = model.predict([4]) # 새로운 x값을 predcit한 결과 
print('4의 예측값 : ', result)

'''
loss :  4.263256414560601e-14
4의 예측값 :  [[3.9999993]]
'''