import torch
import torch.nn as nn

# 假设词汇表包括以下单词和一个填充标记
vocab = {"hello": 1, "world": 2, "how": 3, "are": 4, "you": 5, "": 0}

# 假设我们有以下文本数据（批量大小为2，长度不一样的序列）
text_data = [["hello", "world", "how"],
             ["are", "you"]]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, vocab, padding_idx):
        self.text_data = text_data
        self.vocab = vocab
        self.padding_idx = padding_idx

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        # 将文本序列转换为索引序列
        indexed_sequence = [self.vocab[word] for word in self.text_data[index]]
        return indexed_sequence


# 创建数据集对象
padding_idx = vocab[""]  # 填充标记的索引
dataset = TextDataset(text_data, vocab, padding_idx)

# 创建数据加载器
batch_size = 2
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化嵌入层并添加随机的嵌入向量值
input_dim = 5  # 嵌入维度
embedding = nn.Embedding(len(vocab), input_dim, padding_idx=padding_idx)

# 打印嵌入后的向量
for batch in data_loader:
    indexed_sequences = batch  # 获取一个批次的索引序列

    # 将索引序列转换为嵌入表示
    embedded_sequences = embedding(torch.tensor(indexed_sequences))

    print("Embedded Sequences:")
    print(embedded_sequences)
