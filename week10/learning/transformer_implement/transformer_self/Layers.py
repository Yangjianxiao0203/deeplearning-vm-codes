import torch.nn as nn
import torch
import torch.nn.functional as F

class MutiHeadAttention(nn.Module):
    def __init__(self,n_head,d_model,d_k,d_v,dropout=0.1):
        super(MutiHeadAttention,self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model,n_head*d_k,bias=False)
        self.w_k = nn.Linear(d_model,n_head*d_k,bias=False)
        self.w_v = nn.Linear(d_model,n_head*d_v,bias=False)

        self.fc_out = nn.Linear(n_head*d_v,d_model,bias=False)

        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self,q,k,v,mask=None):
        d_k,d_v,n_head = self.d_k,self.d_v,self.n_head
        batch_size = q.size(0)
        query_len,key_len,value_len = q.size(1),k.size(1),v.size(1)

        residual = q

        q = self.w_q(q).view(batch_size,query_len,n_head,d_k) # b x lq x n x dk
        k = self.w_k(k).view(batch_size,key_len,n_head,d_k) # b x lk x n x dk
        v = self.w_v(v).view(batch_size,value_len,n_head,d_v) # b x lv x n x dv

        qk = torch.einsum('bqnd,bknd -> bnqk',q,k) # b x n x lq x lk
        if mask is not None:
            qk = qk.masked_fill(mask == 0,-1e9)
        qk = qk / (d_k ** 0.5)
        qk = F.softmax(qk,dim=-1) # b x n x lq x lk
        qk = self.dropout(qk) # b x n x lq x lk

        qkv = torch.einsum('bnqk,bvnd -> bnqd',qk,v) # b x n x lq x dv
        output = qkv.view(batch_size,query_len,-1) # b x lq x n*dv
        output = self.fc_out(output) # b x lq x d_model
        output = self.dropout(output) # b x lq x d_model
        output = self.layer_norm(output + residual) # b x lq x d_model
        return output



class FeedForward(nn.Module):
    def __init__(self,d_model,d_inner,dropout=0.1):
        super(FeedForward,self).__init__()

class EncoderLayer(nn.Module):
    def __init__(self,d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        '''
        Args:
            d_model: 词向量维度
            d_inner: 前馈神经网络隐藏层维度
            n_head: 多头注意力头数
            d_k: 多头注意力中每个头的词向量维度
            d_v: 多头注意力中每个头的词向量维度
        '''
        super(EncoderLayer,self).__init__()
        self.attention_self = MutiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        self.ffn = FeedForward(d_model,d_inner,dropout=dropout)
    def forward(self,x,mask = None):
        '''
        x: batch_size * seq_len * d_model
        '''
        return x

if __name__ == "__main__":
    # 设置随机数生成器的种子，以确保结果的可重复性
    torch.manual_seed(42)

    # 初始化多头注意力模块
    model = MutiHeadAttention(n_head=8, d_model=64, d_k=8, d_v=8)

    # 创建输入数据
    q = torch.rand(2, 5, 64)  # Batch size=2, Sequence length=5, Feature dimension=64
    k = torch.rand(2, 10, 64)  # Sequence length=10 for k
    v = torch.rand(2, 10, 64)  # Sequence length=10 for v

    # Forward pass
    output = model(q, k, v)

    print(output.shape)  # 应该输出torch.Size([2, 5, 64])
