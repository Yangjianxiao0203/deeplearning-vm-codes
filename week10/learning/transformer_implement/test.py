import numpy as np

x= np.array([
    [1,2,-3,4],
    [-1,1,1,3]
])  # 2*4
y=np.array([4,2,-8,2]) #1*4

x_trans = x - np.mean(x,axis=1,keepdims=True)

print(x_trans @ x_trans.T /4)
