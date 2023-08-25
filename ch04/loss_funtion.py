import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7    # log(0) : -INF 방지
    return -np.sum(t * np.log(y+delta))

import sys, os
sys.path.append(os.pardir)  # parent directory
from data.mnist import load_mnist  # data에 있는 mnist 파일 속 load_mnist() import

# 1차원 넘파이 배열로 flatten
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]   # batch_mask : index
t_batch = t_train[batch_mask]

def cross_entropy_error_minibatch_binary(y, t):     # 정답 : 0 or 1
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_minibatch_digit(y, t):      # 정답 : 0 ~ 9
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

