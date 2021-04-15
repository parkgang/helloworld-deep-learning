#%%
import tensorflow as tf
import keras as k

print("텐서플로우 버전 = ", tf.__version__)
print("케라스 버전 = ", k.__version__)
#%%
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# %%
