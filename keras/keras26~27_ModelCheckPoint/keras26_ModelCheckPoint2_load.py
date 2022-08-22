from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# #2. 모델링 
# model = Sequential()
# model.add(Dense(40, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))


#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam') 

# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_2_MCP.hdf5')

# hist = model.fit(x_train,y_train,epochs=50, batch_size=8,validation_split=0.25, callbacks=[es,mcp]) 

# print("-------------------------------------------")
# print(hist.history['val_loss']) # hist.history var_loss키 값의 value들을 출력해준다.
# print("-------------------------------------------")


#model.save("./_save/keras26_2_save_MCP.h5")
# restore_best_weights True 했을때
#model = load_model('./_ModelCheckPoint/keras26_2_MCP.hdf5')
#model = load_model('./_save/keras26_2_save_MCP.h5')
# 확인해봤는데 값이 똑같이 나온다.

# restore_best_weights False 했을때
model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')     # 향상된 값이 나온다. 왜 why? 와 조금이 아니라 엄청 상향된값 나오네 차이 심하네.
#model = load_model('./_save/keras26_1_save_MCP.h5')            # 추가로 그러면 save모델에 저장되는 값은 es의 영향을 받는다는 뜻인거 같네. <- 마지막값을 저장했다.
                                                                # 반환에서 돌아오는 값은 es가 주는값. 그리고 프로그램이 주는 loss값 r2스코어 모두 es에서 주는 값. <- 마지막값
                                                                # mcp는 es와 model.save와 상관없이 독자적으로 값을 따로 저장하고 파일까지 저장한다.
                                                                # mcp file_save는 mcp가 주는값을 저장하고 model.save와 프로그램이 출력해주는 값은 es가 주는 값을 출력.
#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

#loss :  30.479597091674805
#r2스코어 :  0.6353370636633953         mcp값 : 0.64

# loss :  32.29319381713867
# r2스코어 :  0.6136389237537831        mcp값 : 0.71