#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#%%
# MNIST(케라스에서 지원하는 데이터 예제 셋) 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#%%
# reshape함수로 2차원 구조의 텐서를 1차원 구조로 변환
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

#%%
# float32 데이터형으로 변환하고 0 255 범위를 0 1 범위로 정규화
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

#%%
# 레이블을 원핫 코드로 변환
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

#%%
# 신경망 구조 설계
n_input = 784
n_hidden = 1024
n_output = 10

mlp=Sequential()

mlp.add(Dense(units=n_hidden, 
              activation='tanh', 
              input_shape=(n_input,),
              kernel_initializer='random_uniform',
              bias_initializer='zeros'))

mlp.add(Dense(units=n_output, 
              activation='tanh', 
              kernel_initializer='random_uniform',
              bias_initializer='zeros'))

#%%
# 신경망 학습

# mean_squared_error 손실 함수 MSE 사용
# => 신경망이 학습할 수 있도록 해주는 지표. 머신러닝 모델의 출력값과 사용자가 원하는 출력값의 차이, 
#   즉 오차를 말함. 이 손실함수 값을 최소화되도록 하는 가중치와 편향을 찾는 것이 바로 학습이다.
#   일반적인 손실함수로 평균 제곱 오차(MSE)나 교차 엔트로피 오차(CEE)를 사용합니다.
# Adam 옵티마이저로 Adam 사용(Adaptive momentum estimation)
# => 옵티마이저 : 학습속도를 빠르고 안정적이게 하는것
#    가장 많이 사용하고 검증된 옵티마이저 Adam
mlp.compile(loss='mean_squared_error', 
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy'])

# 학습 도중에 발생한 정보를 hist 객체에 저장해 둠 (시각화 활용)
hist=mlp.fit(x_train,y_train,
             batch_size=128, 
             epochs=30,
             validation_data=(x_test,y_test),
             verbose=2)

#%%
# 학습된 신경망으로 예측
# evaluate 파라미터로 학습 데이터와 학습레이블을 넣어줍니다. 결과는 [손실, 정확도]로 반환합니다.
res=mlp.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

# %%
