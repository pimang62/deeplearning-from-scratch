import numpy as np
import matplotlib.pyplot as plt

# value
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# array
def step_function(x):
    y = x > 0   # [False, True, True]
    return y.astype(np.int64)  # [0, 1, 1]

print(step_function(x=np.array([-1.0, 1.0, 2.0])))

x = np.arange(-5.0, 5.0, 0.1)  # -5.0~5.0 (간격 0.1)
y1 = step_function(x)
plt.plot(x, y1)
plt.ylim(-0.1, 1.1)
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y2 = sigmoid(x)
plt.plot(x, y2)
plt.ylim(-0.1, 1.1)
plt.show()

# step & sigmoid
plt.plot(x, y1, label='step function')
plt.plot(x, y2, linestyle='--', label='sigmoid')
plt.show()

def ReLU(x):
    y = np.maximum(0, x)
    return y

print(ReLU(x=[-5.0, 0.4, 5.0]))

plt.plot(x, ReLU(x))
plt.show()

def identify_function(x):
    return x

# 3-floor neuron
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forword(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(W1.T, x) + b1   # 1st
    z1 = sigmoid(a1)
    a2 = np.dot(W2.T, z1) + b2  # 2nd
    z2 = sigmoid(a2)
    a3 = np.dot(W3.T, z2) + b3  # 3rd
    y = identify_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forword(network, x)
print(y)  # [0.31682708 0.69627909]

# overflow
def softmax(a):
    y = np.exp(a) / np.sum(np.exp(a))
    return y

print(softmax(a = np.array([0.3, 2.9, 4.0])))
print(softmax(a = np.array([1010, 1000, 990])))  # overflow

def softmax(a):
    c = np.max(a)  # max값 빼주기
    y = (np.exp(a-c)) / (np.sum(np.exp(a-c)))
    return y

print(softmax(a = np.array([0.3, 2.9, 4.0])))  # 식 변화 없음
print(softmax(a = np.array([1010, 1000, 990])))