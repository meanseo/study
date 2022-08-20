from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
from pandas import get_dummies
from tensorflow.keras.layers import Dense, Input

datasets = load_iris()
x = datasets.data
y = datasets.target

y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)      
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    

'''
model = Sequential()
model.add(Dense(70, activation='linear', input_dim=4))    
model.add(Dense(55))   
model.add(Dense(40,activation='relu'))
model.add(Dense(25))
model.add(Dense(10,activation='relu'))
model.add(Dense(3, activation='softmax'))  
'''
input = Input(shape=(4,))
d1 = Dense(70, activation='linear')(input)
d2 = Dense(55)(d1)
d3 = Dense(40, activation='relu')(d2)
d4 = Dense(25)(d3)
d5 = Dense(10, activation='relu')(d4)
output = Dense(3, activation='softmax')(d5)

model = Model(input, output)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])  
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.11111111, callbacks=[es])

model.save("./_save/keras25_4_save_iris.h5")
#model = load_model("./_save/keras25_4_save_iris.h5")

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)