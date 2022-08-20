from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
from pandas import get_dummies

#1.데이터 로드 및 정제

datasets = load_iris()
x = datasets.data
y = datasets.target

print(np.unique(y)) # [0 1 2] -> 다중 분류
y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   #어떤 스케일러 사용할 건지 정의
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)      
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    

#2. 모델구성,모델링
model = Sequential()
model.add(Dense(70, activation='linear', input_dim=4))    
model.add(Dense(55))   
model.add(Dense(40,activation='relu')) #
model.add(Dense(25))
model.add(Dense(10,activation='relu')) #
model.add(Dense(3, activation='softmax'))  

model.summary()

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 
es = EarlyStopping  
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.11111111, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

'''
결과정리            일반레이어                      relu
NoScaler  
loss :          0.0033079693093895912       0.0010957516497001052
accuracy :      1.0                         1.0

MinMax
loss :          0.007345080375671387        0.0005389913567341864
accuracy :      1.0                         1.0

Standard
loss :          0.0014144571032375097       0.011362032033503056
accuracy :      1.0                         1.0

Robust
loss :          0.0014624781906604767       0.0034313490614295006
accuracy :      1.0                         1.0

MaxAbs
loss :          0.0035078583750873804       0.0006933937547728419
accuracy :      1.0                         1.0
'''