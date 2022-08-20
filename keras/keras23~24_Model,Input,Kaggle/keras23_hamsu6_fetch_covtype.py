from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pandas import get_dummies

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

'''
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=54))    
model.add(Dense(80))
model.add(Dense(60 ,activation="relu")) #
model.add(Dense(40))
model.add(Dense(20 ,activation="relu")) #  
model.add(Dense(7, activation='softmax')) 
'''

input1 = Input(shape=(54,))
dense1 = Dense(100,activation="relu")(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(60,activation="relu")(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(7,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10000,validation_split=0.11111111, callbacks=[es])

model.save("./_save/keras25_6_save_covtype.h5")
#model = load_model("./_save/keras25_6_save_covtype.h5")

loss = model.evaluate(x_test,y_test)
print(loss)