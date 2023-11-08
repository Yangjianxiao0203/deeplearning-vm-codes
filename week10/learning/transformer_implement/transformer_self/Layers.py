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
            print("mask shape: ",mask.shape)
            print("qk shape: ",qk.shape)
            qk = qk.masked_fill(mask == 0,-1e9)
        qk = qk / (d_k ** 0.5)
        qk = F.softmax(qk,dim=-1) # b x n x lq x lk
        qk = self.dropout(qk) # b x n x lq x lk

        attention = qk

        qkv = torch.einsum('bnqk,bvnd -> bnqd',qk,v) # b x n x lq x dv
        output = qkv.view(batch_size,query_len,-1) # b x lq x n*dv
        output = self.fc_out(output) # b x lq x d_model
        output = self.dropout(output) # b x lq x d_model
        output = self.layer_norm(output + residual) # b x lq x d_model
        return output, attention



class FeedForward(nn.Module):
    def __init__(self,d_model,d_inner,dropout=0.1):
        super(FeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_inner)
        self.w_2 = nn.Linear(d_inner,d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)
    def forward(self,x):
        '''
        x: batch_size * seq_len * d_model
        '''
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        y_pred = self.layer_norm(x)
        return y_pred

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
        output,attention = self.attention_self(x,x,x,mask=mask) #ouput: batch_size * seq_len * d_model, attention: batch_size * n_head * seq_q * seq_k
        output = self.ffn(output) # batch_size * seq_len * d_model

        return output,attention

class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_inner,n_head,d_k,d_v,dropout=0.1):
        '''
        Args:
            d_model: 词向量维度
            d_inner: 前馈神经网络隐藏层维度
            n_head: 多头注意力头数
            d_k: 多头注意力中每个头的词向量维度
            d_v: 多头注意力中每个头的词向量维度
        '''
        super(DecoderLayer,self).__init__()
        self.attention_self = MutiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        self.attention_target = MutiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        self.ffn = FeedForward(d_model,d_inner,dropout=dropout)
    def forward(self,x,target,encoder_decoder_mask = None,self_mask = None):
        '''
        x: batch_size * seq_len * d_model from encoder
        target: batch_size * seq_len * d_model from decoder
        '''
        target,target_attention = self.attention_self(target,target,target,mask=self_mask) # batch_size * seq_len * d_model
        # q from decoder, k,v from encoder
        x,encoder_decoder_attention = self.attention_target(target,x,x,mask=encoder_decoder_mask) # batch_size * seq_len * d_model
        x = self.ffn(x) # batch_size * seq_len * d_model
        return x,target_attention,encoder_decoder_attention

if __name__ == "__main__":
    # 设置随机数生成器的种子，以确保结果的可重复性
    torch.manual_seed(42)

    # 初始化EncoderLayer模块
    encoder_layer = EncoderLayer(d_model=64, d_inner=256, n_head=8, d_k=8, d_v=8)

    # 创建输入数据
    x = torch.rand(2, 5, 64)  # Batch size=2, Sequence length=5, Feature dimension=64

    # 假设我们有一个mask，这里我们使用全1的mask表示没有位置被遮挡。
    # 在真实场景中，你可能会有0在某些位置来遮挡某些词。
    mask = torch.ones(2, 1, 1, 5)  # 这里的mask大小是: [batch_size, num_heads, query_len, key_len]

    # Forward pass
    output, attention = encoder_layer(x, mask=mask)

    print("Output shape:", output.shape)  # 应该输出torch.Size([2, 5, 64])
    print("Attention shape:", attention.shape)  # 应该输出torch.Size([2, 8, 5, 5])

