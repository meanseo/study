# 과적합 예제
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping


#1 데이터 정제작업
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping  #정의를 해줘야 쓸수있다. 
es = EarlyStopping(monitor="val_loss", patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# 멈추는 시점은 최소값. 발견 직후 patience 값에서 멈춘다. 
# patience: Number of epochs with no improvement after which training will be stopped.
# restore_best_weights : False일 경우 마지막training이 끝난 후의 weight값을 저장/ True라면 training이 끝난 후 값이 가장 좋았을때의 weight로 복원

# 숙제.
# 단 이때 제공되는 weight의 값은 최저점 val_loss에서의 weight값일까 아니면 마지막 trainig이 끝난후의 weight값일까? 
# restore_best_weights=True 로 했기 때문에 최저 val_loss에서의 w 값? Q

start = time.time()
hist = model.fit(x_train,y_train,epochs=34,
                batch_size=1,validation_split=0.25, callbacks=[es]) 
end = time.time() - start
#print("걸린시간 : ", round(end, 3), '초')

#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
# print(hist.history['val_loss'])
print('loss : ', loss)

# y_predict = model.predict(x_test)
# print("최적의 로스값 : ", y_predict)

# r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
# print('r2스코어 : ', r2)


# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()