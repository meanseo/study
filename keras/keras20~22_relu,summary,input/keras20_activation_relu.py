from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

#2. 모델구성 activation -> 레이어의 결과값을 다음으로 전달할때 그 값을 한정시켜준다.
# 밑바닥 책 -> 입력 신호의 총합을 출력 신호로 변환하는 함수
# 얼마나 출력할 지 결정 후 층을 쌓아 비선형을 표현할 수 있도록 해줌
model = Sequential()
model.add(Dense(10, input_dim=1)) 
model.add(Dense(5, activation="relu"))  # ReLU -> 음수는 0으로 만들고 양수는 그대로 가져오게 해주는 함수.
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(3))    
model.add(Dense(1)) 


#3. 컴파일, 훈련     
model.compile(loss='mse', optimizer='adam') 

model.fit(x, y, epochs=50, batch_size=1) 


#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([4]) 
print('4의 예측값 : ', result)