import numpy as np
class DiyModel:
    '''
    x: (n,m): m:feture number, n:sample number
    y: (n,k) 多变量回归任务
    w: (m,k) 多变量回归任务
    '''
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        y_pred = np.dot(x,self.weight)
        y_pred = self.diy_sigmoid(y_pred)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    #sigmoid
    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #手动实现mse，均方差loss
    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true)) / len(y_pred)

    #手动实现梯度计算
    def calculate_grad(self, y_pred, y_true, x):
        #反向过程
        # 均方差函数 (y_pred - y_true) ^ 2 / n 的导数 = 2 * (y_pred - y_true) / n
        grad_loss_sigmoid_wx = 2/len(x) * (y_pred - y_true) # (n,k)
        print('grad_loss_sigmoid_wx: ', grad_loss_sigmoid_wx.shape)

        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y)
        grad_sigmoid_wx_wx = y_pred * (1 - y_pred) # (n,k)
        print('grad_sigmoid_wx_wx: ', grad_sigmoid_wx_wx.shape)
        # wx对w求导 = x
        grad_wx_w = x[:,:,None] # (n,m,1)
        print('grad_wx_w: ', grad_wx_w.shape)
        #导数链式相乘
        dl_dz = (grad_loss_sigmoid_wx * grad_sigmoid_wx_wx)[:,None,:] # (n,1,k)
        print('dl_dz: ', dl_dz.shape)
        grad_before = dl_dz * grad_wx_w # (n,1,k) * (n,m,1) = (n,m,k)
        print('grad before x: ', grad_before.shape)
        #求和
        grad = np.sum(grad_before, axis=0) # (m,k)
        print('grad: ', grad.shape)
        return grad

import torch

class TorchModel(torch.nn.Module):
    def __init__(self, weight):
        super(TorchModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(torch.mm(x, self.weight))

def compare_gradients(diy_model, x, y):
    # 获取diy模型的梯度
    y_pred_diy = diy_model.forward(x)
    diy_grad = diy_model.calculate_grad(y_pred_diy, y, x)

    # 定义PyTorch模型
    torch_model = TorchModel(diy_model.weight)
    criterion = torch.nn.MSELoss()

    x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y, dtype=torch.float32)
    y_pred_torch = torch_model(x_torch)
    loss = criterion(y_pred_torch, y_torch)
    loss.backward()

    # 获取PyTorch模型的梯度
    torch_grad = torch_model.weight.grad

    # 比对梯度
    diff = np.abs(diy_grad - torch_grad.detach().numpy())
    print(f"Maximum difference in gradients: {diff.max()}")
    if np.allclose(diy_grad, torch_grad.detach().numpy(), atol=1e-6):
        print("The gradients are almost identical!")
    else:
        print("The gradients differ!")
    print("diy grad_\n", diy_grad)
    print("torch grad_\n", torch_grad.detach().numpy())

# Test the comparison function
x_np = np.random.random((2, 3))
y_np = np.random.random((2, 2))
weight_np = np.random.random((3, 2))
diy_model_instance = DiyModel(weight_np)
compare_gradients(diy_model_instance, x_np, y_np)


#test
# if __name__ == '__main__':
#     weight = np.random.random((9, 4))
#     diy_model = DiyModel(weight)
#     # random input x:
#     x = np.random.random((8, 9))
#     y= np.random.random((8, 4))
#
#     #epoch
#     for i in range(10):
#         #forward
#         y_pred = diy_model.forward(x)
#         #backward
#         grad = diy_model.calculate_grad(y_pred, y, x)
#         #update weight
#         weight = weight - 0.01 * grad
#         diy_model.weight = weight
#         print('-------------------')