import sys, os
sys.path.append('C:/Users/yelin/deeplearning/Scratch')
import numpy as np
from deep_learning_from_scratch.common.functions import softmax, cross_entropy_error
from deep_learning_from_scratch.common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
net = simpleNet()
print(net.W)  # 가중치  
'''
[[-0.60635139 -0.21546264  2.24012995]
 [-1.04850789  1.43116693  1.09508914]]
'''

x = np.array([0.6, 0.9])
p = net.predict(x)  # y 추정치
print(p)
'''
[-1.30746794  1.15877265  2.3296582 ]
'''

print(np.argmax(p))  # 최댓값의 인덱스
'''
2
'''

t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))
'''
0.2899943509967014
'''

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)  # 기울기
print(dW)
'''
[[ 0.22589968  0.02539612 -0.2512958 ]
 [ 0.33884952  0.03809419 -0.37694371]]
'''

f = lambda w : net.loss(x, t)  # 위와 동일 함수
dW = numerical_gradient(f, net.W)
print(dW)
'''
[[ 0.22589968  0.02539612 -0.2512958 ]
 [ 0.33884952  0.03809419 -0.37694371]]
'''