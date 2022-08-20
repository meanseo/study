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

#model.summary()

#model.save("./_save/keras25_1_save_model.h5")
#model.save_weights("./_save/keras25_1_save_weights.h5")
# model.load_weights('./_save/keras25_1_save_weights.h5')   
# loss :  10535.8427734375
# r2스코어 :  -125.05255477395185



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

model.fit(x_train,y_train,epochs=50, batch_size=1,validation_split=0.25) 

#model.save("./_save/keras25_3_save_model.h5")
#model.save_weights("./_save/keras25_3_save_weights.h5")
model.load_weights('./_save/keras25_3_save_weights.h5')
# loss :  29.492328643798828
# r2스코어 :  0.6471489918709401

'''
가중치만 저장하기 때문에, 모델 architecture를 동일하게 만들어야 함.
이미 모델 architecture를 알고 있을 때만 사용 가능.
'''

# save_weights, load_weights는 일반 save와 다르게 model = Sequential()과 model.compile()해줘야 사용이 가능하다
# fit단계 전에 하냐 후에 하냐에 따라 차이가 있지만 후에 쓰는게 바른 방법이고 그래야 값이 저장된다.

#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)