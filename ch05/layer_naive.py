class Mullayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
        
    def backward(self, dout):
        dx = dout * self.y  # x change to y
        dy = dout * self.x  # y change to x
        
        return dx, dy
    
class Addlayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy
        
