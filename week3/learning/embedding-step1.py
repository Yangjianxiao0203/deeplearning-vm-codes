import torch.nn as nn
import torch
'''
Embedding矩阵是可训练的参数，一般会在模型构建时随机初始化
也可以使用预训练的词向量来做初始化，此时也可以选择不训练Embedding层中的参数

输入的整数序列可以有重复，但取值不能超过Embedding矩阵的列数

核心价值：将离散值转化为向量
'''

'''
当你初次实例化一个nn.Embedding层时，其权重通常是随机初始化的。在这种随机初始化状态下，每行（代表每个词的嵌入）的向量大概率上是线性独立的。

在模型训练过程中，这些嵌入向量会通过梯度下降进行更新以最小化某个损失函数。在训练后，嵌入向量可能不再是线性独立的。
这是因为模型可能发现某些词在语义上相似或共享某些特性，因此它们的嵌入可能会在某些维度上靠得更近。
'''

# 假设我们词表有6个字符，每个字符希望被嵌入到5维的向量中，用5维的向量去表示它
num_embeddings = 6
embedding_dim = 5
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
print("随机初始化权重")
print(embedding_layer.weight) # 6*5的矩阵
print("################")

# 构造词表
vocab = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
}

# 将字符串转换为序列
def str_to_sequence(string, vocab):
    res=[]
    for s in string:
        if s not in vocab:
            raise Exception("unknow character")
        res.append(vocab[s])
    return res

test_strings = [
    "abcde",
    "ddccb",
    "fedab",
]

sequences = [str_to_sequence(string, vocab) for string in test_strings]
print("test_strings convert into sequences: ",sequences)

# 将序列转换为张量
tensor = torch.LongTensor(sequences) # 3*5 即num * len (string) 每一个字串都是一个矩阵
print("sequences convert into tensor: ",tensor)

res = embedding_layer(tensor) # 3*5 *5 即num * len (string) * embedding_dim,根据每个字符对应的5维向量，把它取出来
print("embedding result: ",res)
print("embedding result shape: ",res.shape)