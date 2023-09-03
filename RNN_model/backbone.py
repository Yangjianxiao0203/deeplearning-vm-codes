import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,batch_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        # x to h
        self.xh = nn.Linear(input_size,hidden_size)
        # h to h
        self.hh = nn.Linear(hidden_size,hidden_size)
        # h to output
        self.hy = nn.Sequential(
            nn.Linear(hidden_size,output_size),
            nn.Softmax()
        )
    
    def forward(self,x,h):
        hidden = torch.tanh(self.xh(x) + self.hh(h))
        output = self.hy(hidden)
        
        return output,hidden