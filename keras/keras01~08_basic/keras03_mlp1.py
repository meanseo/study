import numpy as np
from tensorflow.keras import models 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from icecream import ic

# mlp : multi layer perceptron

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])   # 10행 2열의 데이터
print(x)
# # ic(x.shape) # shape = (2, 10)
# y = np.array([11,12,13,14,15,16,17,18,19,20])
# x = x.T # 배열의 행과 열을 바꿈
# # ic(x)
# ic(x.shape)
# '''
# ic| x: array([[ 1. ,  1. ],
#               [ 2. ,  1.1],
#               [ 3. ,  1.2],
#               [ 4. ,  1.3],
#               [ 5. ,  1.4],
#               [ 6. ,  1.5],
#               [ 7. ,  1.6],
#               [ 8. ,  1.5],
#               [ 9. ,  1.4],
#               [10. ,  1.3]])
# '''

# # x = np.transpose(x)  # 위와 같은 기능
# # ic(x.shape) # shape = (2, 10)

# # x = x.reshape(10, 2)
# # ic(x)
# # 변환은 되는데 print해서 확인해보면 짝이 다름
# '''
# ic| x: array([[ 1. ,  2. ],
#               [ 3. ,  4. ],
#               [ 5. ,  6. ],
#               [ 7. ,  8. ],
#               [ 9. , 10. ],
#               [ 1. ,  1.1],
#               [ 1.2,  1.3],
#               [ 1.4,  1.5],
#               [ 1.6,  1.5],
#               [ 1.4,  1.3]]) 
# '''
# # transpose 와 reshape의 차이점 transpose는 변환의 개념이고 , reshape는 데이터를 늘였다 줄였다하면서 형태를 바꿔주는 개념이다.
# # reshape는그래서 원본이 보존된다.

# #2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=2)) # 벡터의 수와 같다. 컬럼,열,특성
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(7))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=500, batch_size=1)

# #4 평가, 예측
# loss = model.evaluate(x,y)
# print('loss : ', loss)
# y_predict = model.predict([[10,1.3]])
# print('[10,1.3]의 예측값 : ', y_predict)    
