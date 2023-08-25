import sys, os
sys.path.append('C:/Users/yelin/deeplearning/Scratch')  # parent directory
from data.mnist import load_mnist  # data에 있는 mnist 파일 속 load_mnist() import

# 1차원 넘파이 배열로 flatten
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)                # 5

print(img.shape)            # (784,)
img = img.reshape(28, 28)   # 원래 이미지 모양으로 변형
print(img.shape)            # (28, 28)

img_show(img)