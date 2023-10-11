import torch
import torch.nn as nn

#TODO : how to compare with the real self-attention block, go to ask teacher
class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads_num):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads_num = heads_num
        self.head_dim = embed_size // heads_num # scaler
        assert (self.head_dim * heads_num == embed_size),"embed_size must be divisible by heads_num"

        self.w_v = nn.Linear(self.embed_size,self.embed_size,bias=False)
        self.w_k = nn.Linear(self.embed_size,self.embed_size,bias=False)
        self.w_q = nn.Linear(self.embed_size,self.embed_size,bias=False)
        self.fc_out = nn.Linear(embed_size,embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        '''
        x: [batch_size, len, embed_size]
        '''
        batch_size, seq_len, embed_size = x.shape
        queries = self.w_q(x).reshape(batch_size,seq_len,self.heads_num,self.head_dim)
        keys = self.w_k(x).reshape(batch_size,seq_len,self.heads_num,self.head_dim)
        values = self.w_v(x).reshape(batch_size,seq_len,self.heads_num,self.head_dim) # [batch_size, seq_len, heads_num, head_dim]

        # nqhd, nkhd -> nqkhd 应该出来是个5维张量，然后对d维度进行累加->nhqk 4维张量
        # enisum 把要用来合并的位置用相同字母表示，其他位置用不同字母表示
        qk = torch.einsum("nqhd,nkhd->nhqk",queries,keys)/(self.head_dim ** 0.5) # [batch_size, heads_num, seq_len, seq_len]
        if mask is not None:
            qk = qk.masked_fill(mask==0,float("-1e20"))
        attention = self.softmax(qk) # [batch_size, heads_num, seq_len, seq_len]
        output = torch.einsum("bhis,bshd->bhid",attention,values) # [batch_size, heads_num, seq_len, head_dim]
        output = output.reshape(batch_size,seq_len,self.heads_num*self.head_dim)
        # print("linear weight: ",self.fc_out.weight)
        # print("linear bias: ",self.fc_out.bias)
        print("before linear: ",output)
        output = self.fc_out(output)
        # print("model output: ",output)
        return output

if __name__ == "__main__":
    embed_size = 8
    heads_num = 4
    model = SelfAttention(embed_size,heads_num)
    batch = 1
    seq_len = 3
    x = torch.rand(batch, seq_len, embed_size)
    out = model(x)
    print(out.shape)  # Expected: [batch, seq_len, embed_size]