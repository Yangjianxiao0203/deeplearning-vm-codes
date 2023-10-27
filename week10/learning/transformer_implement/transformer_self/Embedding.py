import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self,d_hid,n_position=200):
        '''
        Description:
            Positional Encoding: 相当于坐标系上半径为 pos/ 10000^(2i/d_hid) 的圆上的点的坐标, 奇数维度为sin, 偶数维度为cos
            PE(pos,2i) = sin(pos/10000^(2i/d_hid))
            PE(pos,2i+1) = cos(pos/10000^(2i/d_hid))
        Args:
            x: batch_size * seq_len * hidden_size
            d_hid: hidden size
            n_position: max length of seq_len
        '''
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table',self._get_sin_cos_table(d_hid,n_position)) # 固定的表,不需要更新，register_buffer代表backward不更新权重

    def _get_sin_cos_table(self,d_hid, n_position):
            '''
            pos / 10000^(2i/d_hid) = exp(log(pos) - 2i/d_hid * log(10000))
            '''
            pos = torch.arange(0,n_position).unsqueeze(1).float() # n_position * 1
            i = torch.arange(0,d_hid,2).float() # (d_hid/2, )  奇数偶数都是间隔为2
            pos_angles = torch.exp(torch.log(pos) - 2 * i / d_hid * torch.log(torch.tensor(10000.0))) # n_position * (d_hid/2)

            table = torch.zeros(n_position,d_hid) # n_position * d_hid
            table[:,0::2] = torch.sin(pos_angles) # even: 把一半的填充为sin，从0开始，间隔为2
            table[:,1::2] = torch.cos(pos_angles) # odd： 把一半的填充为cos，从1开始，间隔为2
            return table.unsqueeze(0) # 1 * n_position * d_hid

    def forward(self,x):
        '''
        x: batch_size * seq_len * d_hid
        '''
        pos = self.pos_table[:,:x.size(1)].clone() # 1 * seq_len * d_hid
        pos = pos.detach() # 不需要更新权重
        return x + pos  # batch_size * seq_len * d_hid



if __name__ == "__main__":
    torch.manual_seed(42)

    # 创建一个输入张量，大小为(batch_size, seq_len, d_hid)
    x = torch.randn(2, 5, 8)  # 例子：batch_size=2, seq_len=5, d_hid=8

    # 初始化PositionalEncoding模块
    pos_enc = PositionalEncoding(d_hid=8, n_position=10)

    # 通过模块传递输入
    output = pos_enc(x)

    # 打印输出
    print(output)


