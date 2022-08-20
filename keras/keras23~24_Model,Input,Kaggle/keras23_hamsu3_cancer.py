from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical  

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train) #fit_transform -> fit부터 transform까지
x_test = scaler.transform(x_test)

'''
model = Sequential()
model.add(Dense(30, input_dim=30))    
model.add(Dense(25 ,activation='relu'))
model.add(Dense(15 ,activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2, activation='sigmoid')) # 이진 분류니까 시그모이드
model.summary()
'''

input = Input(shape=(30,))
dense1 = Dense(30)(input)
dense2 = Dense(25, activation='relu')(dense1)
dense3 = Dense(15, activation='relu')(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(5)(dense4)
output = Dense(2, activation='sigmoid')(dense5)
model = Model(input, output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', baseline=None,
                   verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.11111, callbacks=[es])

model.save("./_save/keras25_3_save_cancer.h5")
#model = load_model("./_save/keras25_3_save_cancer.h5")

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
