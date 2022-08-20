import numpy as np
from tensorflow.keras.models import Sequential, Model # 함수형모델 Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
x = np.array([range(100), range(301,401), range(1,101)]) 
y = np.array([range(71,81)])
print(x.shape, y.shape)
x = np.transpose(x)
y = np.transpose(y)

#2. 모델구성
input1 = Input(shape=(3,))      # input 입력
dense1 = Dense(10)(input1)      # input1에서 받아 입력으로 사용
dense2 = Dense(9)(dense1)       # dense1에서 받아 입력으로 사용
dense3 = Dense(8, activation='relu')(dense2)    # activaiton적용       
output1 = Dense(1)(dense3)      
model = Model(inputs=input1,outputs=output1)   #함수형 모델 inputs 시작과 outputs 끝을 지정
model.summary()

"""
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3)]               0            input 정보도 표시해준다
_________________________________________________________________
dense (Dense)                (None, 10)                40
dense_1 (Dense)              (None, 9)                 99
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 80
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
"""