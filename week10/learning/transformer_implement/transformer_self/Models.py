import torch.nn as nn
import torch
from .Embedding import PositionalEncoding
from .Layers import EncoderLayer,DecoderLayer
from .utils.Masks import get_pad_mask,get_subsequent_mask

__author__ = "Jianxiao Yang"

class Transformer(nn.Module):
    def __init__(
            self, n_source_vocab, n_target_vocab, source_padding_idx, target_padding_idx,
            d_word_vec=512, d_model=512, d_inner=2048, n_layers = 6, n_head = 8,
            d_k=64,d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,scale_emb_or_prj='none'):
        """
        Transformer Model for Sequence to Sequence tasks.

        Args:
            n_source_vocab (int): Size of source vocabulary.
            n_target_vocab (int): Size of target vocabulary.
            source_padding_idx (int): Index of the source padding token.
            target_padding_idx (int): Index of the target padding token.
            d_word_vec (int, optional): Dimension of word vectors. Defaults to 512.
            d_model (int, optional): Dimension of model representations. Defaults to 512. 即attention出来的尺寸
            d_inner (int, optional): Dimension of the inner FFN layer. Defaults to 2048. 前馈网络的隐藏层尺寸
            n_layers (int, optional): Number of transformer block layers. Defaults to 6.
            n_head (int, optional): Number of attention heads. Defaults to 8.
            d_k, d_v (int, optional): Dimension of the key and value vectors. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            n_position (int, optional): Maximum number of positions. Defaults to 200. 也代表了允许的最大句子长度
            trg_emb_prj_weight_sharing (bool, optional): Whether to share weights between the target embedding and the final logit dense layer. Defaults to True. target的embedding和最后线性输出层权重是否共享
            emb_src_trg_weight_sharing (bool, optional): Whether to share weights between the source and target embeddings. Defaults to True. source和target的embedding是否共享
            scale_emb_or_prj (str, optional): Scale the embedding or projection weights by a constant. Defaults to 'none'. 用一个常数来缩放embedding或者projection的权重
        """
        super(Transformer, self).__init__()
        self.source_padding_idx = source_padding_idx
        self.target_padding_idx = target_padding_idx
        self.n_head = n_head
        assert scale_emb_or_prj in ['emb','prj','none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_projection = scale_emb_or_prj == 'prj' if trg_emb_prj_weight_sharing else False

        self.d_model=d_model

        self.encoder = Encoder(
            n_source_vocab=n_source_vocab, d_word_vec=d_word_vec, n_layers=n_layers, n_heads=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=source_padding_idx,
            dropout=dropout, n_position=n_position, scale_emb=scale_emb
        )
        self.decoder = Decoder(
            n_target_vocab=n_target_vocab, d_word_vec=d_word_vec, n_layers=n_layers, n_heads=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx= target_padding_idx,
            dropout=dropout, n_position=n_position, scale_emb=scale_emb
        )

        self.target_word_projection = nn.Linear(d_model, n_target_vocab, bias=False)

        # 这将使用Xavier均匀初始化（也称为Glorot初始化）来初始化权重矩阵p。
        # Xavier初始化的目的是使每一层的输出的方差应该尽量等于其输入的方差，这有助于在深度网络中获得更好的信号传递。
        # 这种初始化方法是深度学习中常用的方法，尤其在使用ReLU激活函数的场合。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, 'd_model and d_word_vec must be equal'

        if trg_emb_prj_weight_sharing:
            # 如果target的embedding和最后线性输出层权重共享，那么target的embedding和最后线性输出层的维度必须相同
            assert d_model == n_target_vocab, \
                'To share the weight between target word embedding matrix and final logit dense layer,' \
                'the dimension of word embedding matrix and final logit dense layer must be equal.'
            self.target_word_projection.weight = self.decoder.target_word_embedding.weight

        if emb_src_trg_weight_sharing:
            # 如果source和target的embedding共享，那么source和target的embedding的维度必须相同
            assert n_source_vocab == n_target_vocab, \
                'To share the weight between source and target word embeddings, ' \
                'the number of source and target vocabulary must be equal.'
            self.encoder.source_word_embedding.weight = self.decoder.target_word_embedding.weight

    def forward(self,x,target):
        '''
        Args:
            x (tensor): Source sequence. Shape: [batch_size, seq_len]
            target (tensor): Target sequence. Shape: [batch_size, target_seq_len]

        When Running:
            mask: batch_size, n_head, seq_len, seq_len
        '''
        self_attn_mask = get_pad_mask(x,self.source_padding_idx,self.n_head)
        sequence_mask = get_subsequent_mask(target)
        source_mask = self_attn_mask
        target_mask = torch.gt((sequence_mask + get_pad_mask(target,self.target_padding_idx,self.n_head)),0)

        encoder_output,*_ = self.encoder(x,source_mask)
        decoder_output,*_ = self.decoder(target,encoder_output,source_mask,target_mask)

        output = self.target_word_projection(decoder_output) # [batch_size, target_seq_len, n_target_vocab]

        if self.scale_projection:
            output = output * (self.d_model ** -0.5)

        output = output.view(-1,output.size(2)) # [batch_size * target_seq_len, n_target_vocab]
        return output



class Encoder(nn.Module):
    def __init__(self,n_source_vocab,d_word_vec,n_layers,n_heads,d_k,d_v,
                 d_model, d_inner, pad_idx, dropout=0.1,n_position=200, scale_emb=False):
        '''
        Args:
            n_source_vocab (int): Size of source vocabulary.
            d_word_vec (int, optional): Dimension of word vectors. Defaults to 512.
            n_layers (int, optional): Number of transformer block layers. Defaults to 6.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            d_k, d_v (int, optional): Dimension of the key and value vectors. Defaults to 64.
            d_model (int, optional): Dimension of model representations. Defaults to 512. 即attention出来的尺寸
            d_inner (int, optional): Dimension of the inner FFN layer. Defaults to 2048. 前馈网络的隐藏层尺寸
            pad_idx (int): Index of the padding token.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            n_position (int, optional): Maximum number of positions. Defaults to 200. 也代表了允许的最大句子长度
            scale_emb (bool, optional): Scale the embedding weights by a constant. Defaults to False. 用一个常数来缩放embedding的权重
        '''
        super(Encoder, self).__init__()
        # word embedding
        self.source_word_emb = nn.Embedding(n_source_vocab, d_word_vec, padding_idx=pad_idx)
        # position embedding
        self.position_emb = PositionalEncoding(d_word_vec, n_position=n_position)
        # backbone * n_layers
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout) for _ in range(n_layers)
        ])
        # dropout
        self.dropout = nn.Dropout(dropout)
        # scale
        self.scale_emb = scale_emb
        #layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # d_model
        self.d_model = d_model

    def forward(self,x,mask=None, return_attention = True):
        '''
        x: batch_size * seq_len
        '''
        attentions_list = []
        # word embedding
        x = self.source_word_emb(x)
        # scale
        if self.scale_emb:
            x *= self.d_model ** 0.5
        # position embedding
        x = self.position_emb(x)
        #dropout and layer normalization
        x = self.layer_norm(self.dropout(x))

        # backbone
        for encoder_layer in self.layer_stack:
            x, attention = encoder_layer(x, mask)
            if return_attention:
                attentions_list.append(attention)

        return x, attentions_list



class Decoder(nn.Module):
    def __init__(
            self, n_target_vocab, d_word_vec, n_layers, n_heads, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        '''
        Args:
            n_target_vocab (int): Size of target vocabulary.
            d_word_vec (int, optional): Dimension of word vectors. Defaults to 512.
            n_layers (int, optional): Number of transformer block layers. Defaults to 6.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            d_k, d_v (int, optional): Dimension of the key and value vectors. Defaults to 64.
            d_model (int, optional): Dimension of model representations. Defaults to 512. 即attention出来的尺寸
            d_inner (int, optional): Dimension of the inner FFN layer. Defaults to 2048. 前馈网络的隐藏层尺寸
            pad_idx (int): Index of the padding token.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            n_position (int, optional): Maximum number of positions. Defaults to 200. 也代表了允许的最大句子长度
            scale_emb (bool, optional): Scale the embedding weights by a constant. Defaults to False. 用一个常数来缩放embedding的权重
        '''
        super(Decoder, self).__init__()
        self.target_padding_idx = pad_idx
        self.d_model = d_model
        self.scale_emb = scale_emb
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        #embedding
        self.target_word_emb = nn.Embedding(n_target_vocab, d_word_vec, padding_idx=pad_idx)
        #position embedding
        self.position_emb = PositionalEncoding(d_word_vec, n_position=n_position)

        #decoder backbone
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self,x, target,target_mask=None, encoder_decoder_mask=None,return_attns = True):
        '''
        Args:
            x: batch_size * seq_len* d_model  from encoder
            target: batch_size * seq_len    anticipated output
            target_mask: batch_size * n_head * seq_len * seq_len   self_mask for decoder
            encoder_decoder_mask: batch_size * n_head * seq_len * seq_len   encoder_decoder_mask for decoder
            return_attns: bool, whether to return attention matrix
        '''
        self_attentions_list = []
        enc_dec_attentions_list = []

        # target embedding
        target = self.target_word_emb(target) # batch_size * seq_len * d_model
        # scale
        if self.scale_emb:
            target *= self.d_model ** 0.5
        # position embedding
        target = self.position_emb(target)
        # dropout and layer normalization
        target = self.layer_norm(self.dropout(target)) # batch_size * seq_len * d_model

        # backbone
        for decoder_layer in self.layer_stack:
            target, self_attn, enc_dec_attn = decoder_layer(x, target, target_mask, encoder_decoder_mask)
            if return_attns:
                self_attentions_list.append(self_attn)
                enc_dec_attentions_list.append(enc_dec_attn)

        return target, self_attentions_list, enc_dec_attentions_list

if __name__ == "__main__":
    # 设置随机数生成器的种子，以确保结果的可重复性
    torch.manual_seed(42)

    # 定义超参数
    n_source_vocab = 5000
    d_word_vec = 512
    n_layers = 6
    n_heads = 8
    d_k = 64
    d_v = 64
    d_model = 512
    d_inner = 2048
    pad_idx = 0
    dropout = 0.1
    n_position = 200
    scale_emb = True

    # 初始化Encoder模块
    encoder = Encoder(n_source_vocab=n_source_vocab, d_word_vec=d_word_vec, n_layers=n_layers,
                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model,
                      d_inner=d_inner, pad_idx=pad_idx, dropout=dropout,
                      n_position=n_position, scale_emb=scale_emb)

    # 创建模拟的输入数据
    x = torch.randint(0, n_source_vocab, (2, 10))  # Batch size=2, Sequence length=10
    mask = (x != pad_idx).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    # Forward pass
    output, attentions = encoder(x, mask)

    print("Output shape:", output.shape)  # 应该输出torch.Size([2, 10, 512])
    print("Attention shape:", attentions[0].shape)  # 注意，这是一个注意力列表，我们只打印第一层的形状，它应该输出torch.Size([2, 8, 10, 10])

