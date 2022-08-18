# 20~30년전의 보스턴 집값 데이터를 활용하는 예제.
# 오늘 배운 모든 것을 총동원해서 알고리즘 완성해보기 
# train_test set 0.6~0.8 사이 , r2 score 0.8이상 
# r2 score (결정계수): 상관계수의 제곱 / 상관계수와는 다르게 상관 분석이 아닌 회귀 분석에서 사용하는 수치) 
# -> 예측 값과 실제 값의 평가 지표 / 회귀 모델의 성능에 대한 평가 지표
# 상관계수 범위 : -1<=r<=1 / 결정계수의 범위 : 0<=r<=1 --> 제곱이니까
# 1에 가까울수록 해당 선형 회귀 모델이 해당 데이터에 대한 높은 연관성을 가지고 있다고 해석할 수 있음
# r2 = r2_score(y, lr.predict(x_2))
# y는 실제값, ir.predict는 fitting된 선형 회귀 모델을 통해 도출된 예측 값


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic

#1 데이터 정제작업
datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(x)
# ic(y)
# print(x.shape)
# print(y.shape)
# print(datasets.feature_names) # 컬럼,열의 이름들
# print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
print(x[:50])

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)


#2. 모델링 
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train,y_train,epochs=200, batch_size=1)

#4. 평가 , 예측  평가는 말그대로 평가만 해보는것
# fit에서 구해진 y = wx + b에 x_test와 y_test를 넣어보고 그 차이가 loss로 나온다?
#loss = model.evaluate(x_test,y_test)
#print('loss : ', loss)

y_predict = model.predict(x_test) #y의 예측값은 x의 테스트값에 wx + b 

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)

# r2스코어 :  0.6945274298896424