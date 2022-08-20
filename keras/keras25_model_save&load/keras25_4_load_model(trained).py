from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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
# model.summary()
#model.save("./_save/keras25_1_save_model.h5")

#model = load_model("./save/keras25_1_save_model.h5")
model = load_model("./_save/keras25_3_save_model.h5")

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam') 
#모델을 저장을 하고불러왔더라도, fit이랑 evaluate를 다시 하면 값이 바뀌어서 나온다.

#model.fit(x_train,y_train,epochs=50, batch_size=1,validation_split=0.25) 

#model.save("./_save/keras25_3_save_model.h5")


#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)    #model.load에서 불러온 모델에 weight값이 없었따면 에러가 난다.그래서 저장시점이 중요하다?
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)

# r2스코어 :  0.6577537076731692

# model.save기능을 이용해 모델을 저장하면 그 안에 #2 모델링 값부터 #3 fit에 있는 epoch validation split등의 값이 다 저장되는거 같다.
# 그래서 load기능을 이용해서 모델을 불러오면 그거 하나만으로 모델을 돌릴 수 있다. 이제 여기서 저장시점이 중요해지는데
# model.save를 모델 다음에 하면 모델까지만 저장하고 fit다음에 하면 weight까지 저장

