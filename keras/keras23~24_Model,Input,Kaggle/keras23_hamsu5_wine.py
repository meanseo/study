from tensorflow.keras.models import Model
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pandas import get_dummies
from tensorflow.keras.layers import Dense,Input

datasets = load_wine()
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
model.add(Dense(120, activation='linear', input_dim=13))    
model.add(Dense(100 ,activation='relu')) #  
model.add(Dense(80))
model.add(Dense(60 ,activation='relu'))  # 
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))
'''
input1 = Input(shape=(13,))
d1 = Dense(120)(input1)
d2 = Dense(100,activation="relu")(d1)
d3 = Dense(80)(d2)
d4 = Dense(60,activation="relu")(d3)
d5 = Dense(40)(d4)
d6 = Dense(20)(d5)
output1 = Dense(3,activation='softmax')(d6)
model = Model(inputs=input1,outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.1111111111, callbacks=[es])

model.save("./_save/keras25_5_save_wine.h5")
#model = load_model("./_save/keras25_5_save_wine.h5")

loss = model.evaluate(x_test,y_test)
print('loss : ',)