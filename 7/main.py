#%%
import tensorflow as tf
import keras as k

print("텐서플로우 버전 = ", tf.__version__)
print("케라스 버전 = ", k.__version__)
#%%
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cifar import load_batch
from ..utils.data.utils import get_file
from .. import backend as k
import numpy as np
import os

def load_data():
    dirname = 'cifar-10-batches-py'
    origin ='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :], y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)
    
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test_transpose(0, 2, 3, 1)
    
    return (x_train, y_train), (x_test, y_test)
# %%
