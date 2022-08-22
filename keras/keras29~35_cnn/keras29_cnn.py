# 파라미터의 수
# (3, 3) 필터 한개에는 3 x 3 = 9개의 파라미터가 있음(Numpy연산 방식으로 이해)
# 그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이 곱해짐
# 그리고 Conv2D(32, ...) 라면 32는 32개의 필터를 적용하여 다음 층에서는 채널이 총 32개가 되도록 만든다는 뜻
# 여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 32개가 추가로 더해짐

# ex) 3 x 3(필터 크기) x 3 (입력 채널(RGB, 흑백이면 1)) x 32(#출력 채널) + 32(출력 채널 bias) = 896
# model.add(Conv2D(a, kernel_size=(b,c), input_shape=(d, e, f)))
# a = filters or kernel
# Filter와 Kernel은 같음 ex) (b,c) -> kernel_size -> 파라미터 연산할때는 b*c값을 사용함 필터크기.
# d, e, 
# f = channel : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation, MaxPooling2D


model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1 ,padding='same',input_shape=(10,10,1)))   #<-- img를 받기위해 사용. 10은 그 다음레이어로 전달할 값 즉, 출력값.
                                # kernel_size=(2,2)  사진을 2,2로 쪼개서 작업하겠다. # Conv2D 할때는 5,5,1하더라도 1을 입력해야한다. RGB구분 위해.
                                # padding='same'는 겉에 0값으로 둘러싸서 kerner_size로 쪼개도 row,col값을 유지시켜준다. default는 valid. -> 유지시켜주지 않음
                                # same padding 한 마디로 입력 사이즈와 출력 사이즈를 동일하게 유지하도록 패딩을 적용
model.add(MaxPooling2D())       # dropout과 비슷한 개념 conv2d가 knrnel을 이용해서 중첩시키며 특성을 추출해 나간다면 maxpoolig은 픽셀을 묶어서 그중에 가장 큰 값만 뺀다.-> 사이즈 감소
                                # maxpooling는 값을 반으로 계속 줄여나간다. default 2,2=4픽셀당 1개  값. 
model.add(Conv2D(5,(3,3), activation='relu'))
model.add(Conv2D(7,(2,2), activation='relu'))
model.add(Flatten())        #<-- 위에서 넘겨주는 값을 일렬로 쭉 나열해서 1개의 값으로 만들어준다.
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))
model.summary()