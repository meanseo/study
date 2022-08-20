from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

#model.save("./_save/keras25_1_save_model.h5")
model.save_weights("./_save/keras25_1_save_weights.h5")



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

model.fit(x_train,y_train,epochs=50, batch_size=1,validation_split=0.25) 

#model.save("./_save/keras25_3_save_model.h5")
model.save_weights("./_save/keras25_3_save_weights.h5")

#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)