import torch.nn as nn
import torch
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
        assert scale_emb_or_prj in ['emb','prj','none']

        self.d_model=d_model


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

class Decoder(nn.Module):
    def __init__(self):
