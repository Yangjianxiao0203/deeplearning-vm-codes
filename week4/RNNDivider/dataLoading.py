import jieba
import torch
from torch.utils.data import DataLoader


# load vocab
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1 #最小索引是1，0用来填充
    vocab['unk'] = len(vocab) + 1
    return vocab


def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence


def sentence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label


class Dataset:
    def __init__(self, corpus_path, vocab,max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.data = []
        self.load()

    def load(self):
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)  # seq_len
                label = sentence_to_label(line)  # seq_len
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                if len(self.data) > 10000:
                    break

    def padding(self, sequence, label):
        '''
        将文本截断或补齐到固定长度，超过的截断，少的补0
        '''
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab,max_length)
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    return data_loader

def load_data(batch_size, max_length):
    vocab_path = '../data/chars.txt'
    corpus_path = '../data/corpus.txt'
    vocab = build_vocab(vocab_path)
    data_loader = build_dataset(corpus_path, vocab, max_length,batch_size)
    return data_loader,vocab

def main(corpus_path, vocab_path):
    batch_size = 32
    max_length = 20
    vocab = build_vocab(vocab_path)
    data_loader = build_dataset(corpus_path, vocab, max_length,batch_size)

    for i, (sequence, label) in enumerate(data_loader):
        x = sequence
        y = label
        break

if __name__ == '__main__':
    vocab_path = '../data/chars.txt'
    corpus_path = '../data/corpus.txt'
    main(corpus_path, vocab_path)
