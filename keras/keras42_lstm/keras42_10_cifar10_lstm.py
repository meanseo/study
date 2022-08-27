#0.내가쓸 기능들 import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping

# 개별 import
from tensorflow.keras.datasets import cifar10
import numpy as np
from pandas import get_dummies
#1.데이터로드 및 정제

### 1-1.로드영역    데이터 형태를 x,y로 정의해주세요.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

### 1-2. RNN하기위해 shape확인.

#x값 관측.    x의 shape를 기록해주세요.     :
# print(x_train.shape)     (50000, 32, 32, 3) 
# print(x_test.shape)      (10000, 32, 32, 3)

# #y값 관측.    y의 shape를 기록해주세요.     :
# print(y_train.shape)    (50000, 1)  
# print(y_test.shape)     (10000, 1)

#무슨 모델쓸지 판단.    관측값 기록 후 모델판단 : (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] => 다중이
#numpy      
# print(np.unique(y_train,return_counts=True))       
# print(np.unique(y_test,return_counts=True))       

#pandas   
#print(y.value_counts())

# onehotencoding은 벡터 형식의 데이터를 처리하므로 벡터형식으로 reshape
y_train = y_train.reshape(len(y_train))   #(50000,)
y_train = get_dummies(y_train)           #(50000,10)
y_test = y_test.reshape(len(y_test))     #(10000,)
y_test = get_dummies(y_test)              #(10000,10)


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


### 1-4. x의 shape변환      이 모델은 입력 데이터가 4차원이므로 DNN사용시 따로 지정해줘야 합니다.  

# RNN 사용시    3차원으로 펴줌
#x_train = x_train.reshape(len(x_train),384,8)      #len(x)뒤의 영역은 사용자 지정입니다!   DNN모델일 경우 주석처리.
#x_test= x_test.reshape(len(x_test),384,8)      #len(x)뒤의 영역은 사용자 지정입니다!   DNN모델일 경우 주석처리.

# DNN 사용시    2차원으로 펴줌
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

### 1-5. train & test분리 
# x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)


### 1-6. scaler적용. 스킵 가능----------------------------------------------------------------------

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()MinMaxScaler()

#Minmax scaler 꼭 사용하세요!!!
# RNN사용시 
# 자동으로 3차원데이터를 2차원으로 만들어서 스케일링 적용하고 다시 3차원으로 적용해줌.
#x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
#x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)                               

# DNN사용시
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#--------------------------------------------------------------------------------------------------


#2.모델링   각 데이터에 알맞게 튜닝
model = Sequential()
#model.add(SimpleRNN(10,input_shape=(x.shape[1],x.shape[2])   ,return_sequences=True))       # 공백안에 ,activation='relu'도 사용해보세요.
#model.add(LSTM(10,return_sequences=True,activation='relu'))                                 # 두번째, 세번째 줄은 주석처리해서 1개만 사용해보세요.
#model.add(GRU(10,return_sequences=False,activation='relu'))    
model.add(Dense(50,input_dim= x_train.shape[1]))                                             # DNN방식적용시 위의 RNN주석 걸고 위의 1-4에서 두번째 옵션 선택합니다.                
model.add(Dense(64))                                                                         # DNN방식 사용시 model.add(Dropout(0.5)) 복사후 사용.
model.add(Dense(32))
model.add(Dense(16,activation="relu")) #
model.add(Dense(8,activation="relu")) #
model.add(Dense(4))
model.add(Dense(10,activation = 'softmax'))    



#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])    
es = EarlyStopping(monitor="val_loss", patience=1, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train, epochs=10000, batch_size=1000,validation_split=0.2,verbose=1,callbacks=[es])        # batch_size 센스껏 조절!  


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
#                                                                              
#                                                                                       
#                                                                                      
#              