import sys, os
sys.path.append('C:/Users/yelin/deeplearning/Scratch')
import numpy as np
import matplotlib.pyplot as plt
from data.mnist import load_mnist
from twolayerNet import TwoLayerNet

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 하이퍼파라미터
iters_num = 100   # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100    # 미니배치 크기
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 1에폭당 반복 수 : 784 / 100
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, y_batch)     # return grads -> {}
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 학습경과 기록
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {str(train_acc)}, {str(test_acc)}")

# Backend QtAgg is interactive backend. Turning interactive mode on. : error 해결
#plt.ioff()

# 그래프 그리기
loss_x = np.arange(len(train_loss_list))
plt.plot(loss_x, train_loss_list, label='train loss')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(loc='upper right')
plt.show()

markers = {'train': 'o', 'test': 's'}
acc_x = np.arange(len(train_acc_list))
plt.plot(acc_x, train_acc_list, label='train acc')
plt.plot(acc_x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()