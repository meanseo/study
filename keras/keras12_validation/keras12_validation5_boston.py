# 20~30년전의 보스턴 집값 데이터를 활용하는 예제.
# 오늘 배운 모든 것을 총동원해서 알고리즘 완성해보기 
# train_test set 0.6~0.8 사이 , r2 score 0.8이상 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x,y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
# print(x.shape, y.shape)
#(506, 13) (506,)

# 모델
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(170))
model.add(Dense(160))
model.add(Dense(150))
model.add(Dense(130))
model.add(Dense(115))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, validation_split=0.3)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss =',loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2_score = ',r2)