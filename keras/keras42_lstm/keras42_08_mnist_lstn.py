#0.내가쓸 기능들 import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping

# 개별 import
from tensorflow.keras.datasets import mnist
import numpy as np
from pandas import get_dummies
#1.데이터로드 및 정제

### 1-1.로드영역    데이터 형태를 x,y로 정의해주세요.

# a = mnist.load_data() 
# datasets = list(a)    쌤한테 질문 
#x = datasets.data
#y = datasets.target
#x = x_train + x_test
#y = y_train + y_test

(x_train, y_train), (x_test, y_test) = mnist.load_data()


### 1-2. RNN하기위해 shape확인.

#x값 관측.    x의 shape를 기록해주세요.     :
#print(x_train.shape)       #(60000, 28, 28)
#print(x_test.shape)        #(10000, 28, 28)     

#y값 관측.    y의 shape를 기록해주세요.     :
#print(y_train.shape)       #(60000,)
#print(y_test.shape)        #(10000,)

#모델 판별 단계.    y값 관측후 기록 및 판단  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 10개이며,
#numpy      
#print(np.unique(y_train,return_counts=True))      # 5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949 
#print(np.unique(y_test,return_counts=True))       #  980, 1135, 1032, 1010,  982,  892,  958, 1028,  974, 1009
#pandas   
#print(y.value_counts())

y_train = get_dummies(y_train)
y_test = get_dummies(y_test)


### 1-3. 상관관계 분석 후 x칼럼제거.        스킵 가능.------------------------------------------------
#데이터가 np일 경우 pandas import해서 변환후 작업. 원핫인코딩 끄고 작업 후 다시 원핫인코딩해주세요.
# import pandas as pd
# x = pd.DataFrame(x, columns=datasets.feature_names)
# x['ydata'] = y
# #print(x.corr())
# x = x.drop(['','ydata'],axis=1)  # drop시킬 column명 기재.
# #print(x.shape)            # 변경된 칼럼개수 확인.  기재 : 
# #그 이후의 작업 계속해주기 위해 numpy로 변환
# x = x.to_numpy()
#---------------------------------------------------------------------------------------------------


### 1-4. x의 shape변환     이 데이터는 DNN하려면 2차원변환을 따로 해줘야  합니다. <-- 입력데이터가 4차원이기때문

# RNN사용시
x_train = x_train.reshape(len(x_train),98,8)     #len(x)뒤의 영역은 사용자 지정입니다!   DNN모델일 경우 주석처리.
x_test = x_test.reshape(len(x_test),98,8)

# DNN사용시
#x_train = x_train.reshape(len(x_train),784) 
#x_test = x_test.reshape(len(x_test),784)

### 1-5. train & test분리 
#x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)


### 1-6. scaler적용. 스킵 가능----------------------------------------------------------------------

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()MinMaxScaler()

# RNN사용시 
# 자동으로 3차원데이터를 2차원으로 만들어서 스케일링 적용하고 다시 3차원으로 적용해줌.
x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

# DNN사용시
#x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1))
#x_test = scaler.transform(x_test.reshape(len(x_test),-1))
#--------------------------------------------------------------------------------------------------


#2.모델링   각 데이터에 알맞게 튜닝
model = Sequential()
#model.add(SimpleRNN(10,input_shape=(x.shape[1],x.shape[2])   ,return_sequences=True))       # 공백안에 ,activation='relu'도 사용해보세요. 
model.add(LSTM(10,return_sequences=True,activation='relu'))                                  # 윗줄을 주석하고 input shape 넣지않고 바로 실행해도 알아서 모델이 돌아갑니다.
model.add(GRU(10,return_sequences=False,activation='relu'))                                  # 두번째, 세번째 줄은 주석처리해서 1개만 사용해보세요
#model.add(Dense(50,input_dim= x.shape[1]))                                                  # DNN방식적용시 위의 RNN주석 걸고 위의 1-4에서 두번째 옵션 선택합니다.                
model.add(Dense(64))                                                                         # DNN방식 사용시 model.add(Dropout(0.5)) 복사후 사용.
model.add(Dense(32))
model.add(Dense(16,activation="relu")) #
model.add(Dense(8,activation="relu")) #
model.add(Dense(4))
model.add(Dense(10,activation = 'softmax'))    # default = 'linear' 이진분류 = 'sigmoid' , 다중분류 = 'softmax' 



#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    # 회귀모델 = mse, 이진분류 = binary_crossentropy, 다중분류 = categorical_crossentropy, 분류는 ,metrics=['accuracy']
es = EarlyStopping(monitor="val_accuracy", patience=50, mode='max',verbose=1,baseline=None, restore_best_weights=True)  # monitor값 입력하세요
model.fit(x_train,y_train, epochs=10000, batch_size=5000,validation_split=0.2,verbose=1,callbacks=[es])        # batch_size 센스껏 조절  



#4.평가,예측        회귀모델은 r2,  분류모델은 accuracy

loss = model.evaluate(x_test,y_test)

###분류모델일때 주석 해제.
print("----------------------loss & accuracy-------------------------")
print(round(loss[0],4))
print(round(loss[1],4))

### 회귀모델일때 주석 해제.
# print("----------------------loss값-------------------------")
# print(round(loss,4))
# y_predict = model.predict(x_test)

# print("=====================r2score=========================")
# r2 = r2_score(y_test,y_predict)
# print(round(r2,4))


#5.결과 정리 창

#                   DNN                 |             CNN                |               RNN
#loss:                                                                     
#                                                                    
#               DNN + Sc.                           CNN + Sc                        RNN + Sc                              
#loss:                                                                                      
#                                                                                      
#              