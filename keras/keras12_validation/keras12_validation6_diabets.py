from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# R2 0.62이상, train_test set 0.7
#1. 데이터 정제
datasets = load_diabetes()
x = datasets.data
y = datasets.target



x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
k = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2,validation_split=0.11)

#4. 평가 , 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

# r2스코어 :  0.6340421293580923
# r2스코어 :  0.646241289117871  epochs=100, batch_size=1
# r2스코어 :  0.6413384495814317  epochs=100, batch_size=1 모델링 변경