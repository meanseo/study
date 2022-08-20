from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical  

#1.데이터 로드 및 정제
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(np.unique(y)) # [0, 1] =>  one hot encoding

y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train) #fit_transform -> fit부터 transform까지
x_test = scaler.transform(x_test)

#2. 모델구성,모델링
model = Sequential()
model.add(Dense(30, input_dim=30))    
model.add(Dense(25 ,activation='relu'))
model.add(Dense(15 ,activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2, activation='sigmoid')) # 이진 분류니까 시그모이드
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', baseline=None,
                   verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.11111, callbacks=[es])

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

'''
결과정리                일반레이어                      relu

NoScaler 
loss :               0.2484138011932373         0.2484138011932373
accuracy :           0.8947368264198303         0.8947368264198303

MinMax
loss :                                          0.2703065872192383
accuracy :                                      0.9298245906829834

Standard
loss :                                          0.18353988230228424    
accuracy :                                      0.9298245906829834

Robust                                          
loss :                                          0.19624097645282745
accuracy :                                      0.9122806787490845

MaxAbs
loss :             
accuracy :         
''' 