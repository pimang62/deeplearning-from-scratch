import numpy as np
import sys, os
sys.path.append('C:/Users/yelin/deeplearning/Scratch')
from common.functions import softmax, cross_entropy_error

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)    # 0보다 작은 값을 True, np.array 형태
        out = x.copy()      # np.array 형태
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1+np.exp(-x))    
        self.out = out
        
        return out
        
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)         # X (W WT) = Y WT
        self.dW = np.dot(self.x.T, dout)    # (XT X) W = XT Y
        self.db = np.sum(dout, axis=0)
        
        return dx

class SofdtmaxWithLoss:
    def __init__(self):
        self.loss = None    # 손실
        self.y = None       # softmax의 출력
        self.t = None       # 정답 레이블 (one-hot)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
        
        