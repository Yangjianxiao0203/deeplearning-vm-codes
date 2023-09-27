import math
from collections import defaultdict


class NewWordDetect:
    def __init__(self, corpus_path):
        self.max_word_length = 5
        self.word_count = defaultdict(int)  # key:word, value:count
        self.left_neighbor = defaultdict(dict)  # key:word, value:dict, key:char, value:count
        self.right_neighbor = defaultdict(dict)  # key:word, value:dict, key:char, value:count
        self.load_corpus(corpus_path)

    def load_corpus(self, path):
        with open(path, encoding='utf8') as f:
            for line in f:
                sentence = line.strip()
                for word_length in range(1, self.max_word_length):
                    self.ngram_count(sentence, word_length)
        return

    # 按照窗口长度取词,并记录左邻右邻
    def ngram_count(self, sentence, word_length):
        '''
        统计sentence中，长度为word length的字词出现的次数，以及左领右舍是谁
        '''
        sen_len = len(sentence)
        for i in range(sen_len - word_length + 1):
            word = sentence[i:i + word_length]
            self.word_count[word] += 1
            # 记录左邻
            if i - 1 >= 0:
                left = sentence[i - 1]
                self.left_neighbor[word][left] = self.left_neighbor[word].get(left, 0) + 1
            # 记录右舍
            if i + word_length < sen_len:
                right = sentence[i + word_length]
                self.right_neighbor[word][right] = self.right_neighbor[word].get(right, 0) + 1
        return

    '''
    计算pmi，即内部稳定度
    pmi(word) = log(p(word)/p(w1)p(w2)p(w3))  w1,w2,w3是word的每个字
    p(word) = count(word)/相同长度的词的总数
    '''
    def calc_pmi(self):
        word_count_by_length = defaultdict(int)
        for word, count in self.word_count.items():
            word_length = len(word)
            word_count_by_length[word_length] = word_count_by_length.get(word_length, 0) + count
        word_pmi = {}
        for word, count in self.word_count.items():
            p_word = count / word_count_by_length[len(word)]
            p_word_char = 1
            for char in word:
                p_word_char *= self.word_count[char] / word_count_by_length[1]
            # print("current word: {}, p_word: {}, p_char: {}".format(word, p_word, p_word_char))
            word_pmi[word] = math.log(p_word / p_word_char, 10)
        return word_pmi

    def calc_left_and_right_entropy(self):
        word_entropy_left = {}
        for word,count_dict in self.left_neighbor.items():
            word_entropy_left[word] = 0
            total = sum(count_dict.values())
            for char,count in count_dict.items():
                p = count / total
                word_entropy_left[word] += -p * math.log(p,10)
        word_entropy_right = {}
        for word,count_dict in self.right_neighbor.items():
            word_entropy_right[word] = 0
            total = sum(count_dict.values())
            for char,count in count_dict.items():
                p = count / total
                word_entropy_right[word] += -p * math.log(p,10)
        return word_entropy_left,word_entropy_right

    def calc_word_value(self):
        word_values = {}
        pmi = self.calc_pmi()
        left_entropy,right_entropy = self.calc_left_and_right_entropy()
        for word in pmi:
            if "," in word or "，" in word:
                continue
            word_pmi = pmi.get(word)
            le = left_entropy.get(word,1e-3)
            re = right_entropy.get(word,1e-3)
            word_values[word] = word_pmi * le * re

        #sort by value
        word_values = sorted(word_values.items(),key=lambda x:x[1],reverse=True)
        return word_values


if __name__ =='__main__':
    nwd = NewWordDetect('../data/sample_corpus.txt')
    word_values = nwd.calc_word_value()
    print([x for x, c in word_values if len(x) == 2][:10])
    print([x for x, c in word_values if len(x) == 3][:10])
    print([x for x, c in word_values if len(x) == 4][:10])
