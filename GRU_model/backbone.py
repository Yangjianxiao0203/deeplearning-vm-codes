# time sequence: batch, time, feature
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#隐状态初始化: H: (bxh, )
def init_state(batch_size, hidden_size, device):
    return (torch.zeros(batch_size, hidden_size).to(device),)

class GRU(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers=1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        #reset gate
        self.R = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        #update gate
        self.Z = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        # candidate hidden state
        self.H = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        
    def forward(self,x,h):
        concat = torch.cat((x,h), dim=1)
        forget = self.R(concat)
        update = self.Z(concat)
        
        # candidate hidden state
        concat2 = torch.cat((x, torch.mul(h,forget)), dim=1)
        candidate_hidden = self.H(concat2)
        
        candidate = torch.mul(update, h) + torch.mul(1-update, candidate_hidden)
        
        return candidate
        
        
        
        

#main
if __name__=='__main__':
    batch_size = 5
    input_size = 10
    hidden_size = 20
    num_layers = 1 

    gru = GRU(input_size, hidden_size, num_layers)

    x = torch.randn(batch_size, input_size) 
    h = torch.zeros(batch_size, hidden_size)

    c = gru(x, h)

    print(c.shape) # torch.Size([5, 20])