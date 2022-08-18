from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score

'''
# r2_score
회귀모델이 주어진 자료에 얼마나 적합한지를 평가하는 지표
y의 변동량대비 모델 예측값의 변동량을 의미함
0~1의 값을 가지며, 상관관계가 높을수록 1에 가까워짐
r2=0.3인 경우 약 30% 정도의 설명력을 가진다 라고 해석할 수 있음
sklearn의 r2_score의 경우 데이터가 arbitrarily할 경우 음수가 나올수 있음
음수가 나올경우 모두 일괄 평균으로 예측하는 것보다 모델의 성능이 떨어진다는 의미
결정계수는 독립변수가 많아질 수록 값이 커지기때문에, 독립변수가 2개 이상일 경우 조정된 결정계수를 사용해야 함
참고 : https://ltlkodae.tistory.com/19
'''
#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])  

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # mean squared error , 평균제곱오차    
model.fit(x,y,epochs=1000, batch_size=1)

#4. 평가, 예측 
#loss = model.evaluate(x,y) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
#print('loss : ', loss)

y_predict = model.predict(x) #y의 예측값은 x의 테스트값에 wx + b 

r2 = r2_score(y,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)

# r2스코어 :  0.8099232981667172
