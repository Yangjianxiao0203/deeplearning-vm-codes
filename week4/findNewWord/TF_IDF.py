import jieba
import math
import os
from collections import defaultdict

'''
TF-IDF: 词频-逆文档频率
词频：一个词在文档中出现的频率
逆文档频率：一个词在多少篇文档中出现过，对于每一个词，
文章/词出现的文章数： 值越大，说明这个词对某些文章来说专一度更高，只出现在某些文章中，属于专业词汇

计算方法：
tf-idf = tf * log(D/(idf + 1)) 每个词
'''


class TFIDF:
    '''
    dir_path: 语料库路径
    '''

    def __init__(self, dir_path):
        self.corpus = []
        self.path = []

        self.load_corpus(dir_path)  # 加载语料库
        self.divide_corpus()  # 对语料库进行分词
        # tf_list: index 即是文章序号，顺序跟self path一样，每一个元素是个字典，key是词，value是词频
        # idf_dict: key是词，value是出现过这个词的文章数
        self.tf_dict_list, self.idf_dict = self.build_tf_idf_dict()  # 统计tf和idf值
        self.tf_idf_dict_list =  self.calc_tf_idf_dict()

    def load_corpus(self, dir_path):
        '''
        加载语料库
        '''
        for path in os.listdir(dir_path):
            if path.endswith('.txt'):
                path = os.path.join(dir_path, path)
                self.path.append(path)
                with open(path, encoding='utf-8') as f:
                    self.corpus.append(f.read())  # 把整篇文章读进去
        return

    def divide_corpus(self):
        '''
        对语料库进行分词
        '''
        self.corpus = [jieba.lcut(text) for text in self.corpus]
        return

    def build_tf_idf_dict(self):
        '''
        统计tf和idf值
        tf：每一个词在当前文档中出现的频率
        idf：对于每一个词，文章总数/词出现的文章数： 值越大，说明这个词对某些文章来说专一度更高，只出现在某些文章中，属于专业词汇
        '''
        tf_dict_list = [] #index 即是文章序号，顺序跟self path一样，每一个元素是个字典，key是词，value是词频
        idf_dict = defaultdict(set) # key: word, value: set of text_index即文章序号

        for index, words in enumerate(self.corpus):
            tf_dict = defaultdict(int)
            for word in words:
                tf_dict[word] += 1/len(words)
                # if word not in idf_dict and index not in idf_dict[word]:
                if index not in idf_dict[word]:
                    idf_dict[word].add(index)
            tf_dict_list.append(tf_dict)
        idf_dict_freq = {key: len(value) for key, value in idf_dict.items()}
        # save idf_dict_freq
        with open('idf_dict_freq.txt','w',encoding='utf-8') as f:
            for key, value in idf_dict_freq.items():
                f.write('{} {}\n'.format(key,value))
        return tf_dict_list, idf_dict_freq

    def calc_tf_idf_dict(self):
        '''
        计算方法：
        tf-idf = tf * log(D/(idf + 1))
        '''
        tf_idf_dict_list = [] #index 即是文章序号，顺序跟self path一样，每一个元素是个字典，key是词，value是得分
        for index, tf_dict in enumerate(self.tf_dict_list):
            tf_idf_dict = defaultdict(float)
            for word, count in tf_dict.items():
                freq_in_doc = self.idf_dict.get(word,0)
                idf = math.log(len(self.path) / (freq_in_doc + 0.01))
                tf_idf_dict[word] = count * idf
            tf_idf_dict_list.append(tf_idf_dict)

        return tf_idf_dict_list
    def find_top_k_words_in_each_doc(self,k=1):
        '''
        找到每个文档中tf-idf值最大的k个词
        '''
        top_k_words = []
        for index, tf_idf_dict in enumerate(self.tf_idf_dict_list):
            top_k_words.append(sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True)[:k])
        return top_k_words

    def find_word(self,string):
        # return idf_value and tf_value in each doc
        word_idf = self.idf_dict.get(string,0)
        for index, tf_dict in enumerate(self.tf_dict_list):
            tf_value = tf_dict.get(string,0)
            print('doc: {} tf_value: {} idf_value: {}'.format(index,tf_value,word_idf))

def main():
    tf_idf = TFIDF('../data/category_corpus')
    top_k_words = tf_idf.find_top_k_words_in_each_doc(10)
    for words in top_k_words:
        print(words)
    print("---===---" * 50)
    # tf_idf.find_word('\n')


if __name__ == '__main__':
    main()
