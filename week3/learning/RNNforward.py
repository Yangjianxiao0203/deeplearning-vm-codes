import torch
import torch.nn as nn
import numpy as np

class TorchRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(TorchRNN,self).__init__()
        # 把batch size放到了第一维度
        self.layer = nn.RNN(input_size,hidden_size,bias=False,batch_first=True)
    def forward(self,x):
        return self.layer(x)

class DiyModel:
    def __init__(self,wih,whh,hidden_size):
        self.w_ih = wih
        self.w_hh = whh
        self.hidden_size = hidden_size
        self.ht_init = np.zeros(hidden_size)
    def forward(self,x):
        output = []
        ht = self.ht_init
        for xt in x:
            u = np.dot(xt.T,self.w_ih)
            w = np.dot(ht.T,self.w_hh)
            ht_next = np.tanh(u+w)
            output.append(ht_next)
            ht=ht_next
        return np.array(output), ht

x = np.array([[1, 2, 3],
              [3, 4, 5],
              [5, 6, 7]])  #网络输入

hidden_size = 2
torch_model = TorchRNN(3, hidden_size)
w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
print(w_ih, w_ih.shape)
print(w_hh, w_hh.shape)

torch_x = torch.FloatTensor(x)
torch_x= torch_x.unsqueeze(0)
print("torch_x shape: ",torch_x.shape)

output, h = torch_model.forward(torch_x)
print(output.detach().numpy(), "torch模型预测结果")
print(h.detach().numpy(), "torch模型预测隐含层结果")
print("---------------")
diy_model = DiyModel(w_ih.T, w_hh.T, hidden_size)
output, h = diy_model.forward(x)
print(output, "diy模型预测结果")
print(h, "diy模型预测隐含层结果")
