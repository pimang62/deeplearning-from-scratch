from layer_naive import Mullayer

apple = 100
apple_num = 2
tax = 1.1

# layers : 계층들
mul_apple_layer = Mullayer()
mul_tax_layer = Mullayer()

# forward : 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)
'''
220.00000000000003
'''

# backward : 역전파
dprice = 1
dapple_price, dtax = mul_apple_layer.backward(dprice)    # dx, dy
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
'''
220.00000000000003
'''

