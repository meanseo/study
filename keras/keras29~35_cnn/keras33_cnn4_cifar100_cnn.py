from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D,\
                                    GlobalAveragePooling2D,BatchNormalization,LayerNormalization
import numpy as np
from tensorflow.keras.datasets import cifar100 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.optimizers import Adam,Adadelta

#1.데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#scaler =StandardScaler()   #MinMaxScaler()RobustScaler()MaxAbsScaler()

#x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
#x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

#2.모델링
model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu')),model.add(LayerNormalization()),model.add(Dropout(0.5))
model.add(Dense(512,activation='relu')),model.add(LayerNormalization()),model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

#3.컴파일, 훈련
optimizer = Adam(learning_rate=0.0001)  # 1e-4     
lr=ReduceLROnPlateau(monitor= "val_acc", patience = 2, mode='max',factor = 0.1, min_lr=1e-6,verbose=False)
es = EarlyStopping(monitor="val_acc", patience= 3, mode='max',verbose=1,baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras33_cifar100_MCP.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['acc']) 
model.fit(x_train,y_train,epochs=100, batch_size=16,validation_split=0.2, callbacks=[lr,es])#

model.save(f"./_save/keras33_save_cifar100.h5")

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#            기본                       기본+Minmax       기본 + standard
# loss :     3.620971441268921          3.04172852      3.0337114334106445
# accuracy : 0.15029999613761902        0.2642          0.2630000114440918