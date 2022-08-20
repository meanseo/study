import numpy as np
from tensorflow.keras.models import Sequential, Model # 함수형모델 Model
from tensorflow.keras.layers import Dense, Input

# 1. 데이터
x = np.array([range(100), range(301,401), range(1,101)])        # 100,3 인데 현재 3,100
y = np.array([range(71,81)])
print(x.shape, y.shape)     # (3,100) (2,100)
x = np.transpose(x)
y = np.transpose(y)
# print(x.shape, y.shape)     # (100,3) (100,2)
# x = x.reshape(1,10,10,3)
# print(x.shape, y.shape)     # (1, 10, 10, 3) (10, 1)

#2. 모델구성        행의 개수는 같아야 하지만 열의 개수는 달라도 된다. 열 = 특성,피쳐,등등
model = Sequential()
model.add(Dense(10, input_dim=3))       # x는 (100,3)이다   행 무시 (N,3)
#model.add(Dense(9, input_shape=(3,)))    # 차원이 늘어나면 행(x값)을 모두 무시하고 열 columns 값만 쓴다.
# input_shape=(3,) 차원이 하나 뿐인 경우 쉼표가 필요
model.add(Dense(8))
model.add(Dense(2))
model.summary()
'''
dense (Dense)                (None, 10)             40
x 자리가 None인 이유 = 행의 개수는 신경쓰지 않겠다. 상관없다.
'''

# 이미지같은 경우는 3차원 모델이라서 다른 방식으로 모델링 해줘야한다. 
# 가로,세로,그리고 컬러(rgb겹침, tensor) 그리고 여러장이니까 + 1차원해서 총 4차원
