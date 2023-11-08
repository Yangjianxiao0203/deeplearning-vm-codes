import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_schema(path,schema_path,logging):
    '''
    封闭式抽取的schema是事先确定好的，不会预测出这之外的属性/类别/实体，所以需要先构建schema
    这次的数据集只有实体-属性-属性值，
    一般会有三种：
        实体 - 关系 - 实体              姚明–妻子–叶莉
        实体 - 属性 - 属性值         姚明–身高–226cm
        实体 - 标签-标签值     姚明–标签–运动员
    '''

    #if exists(schema_path): return
    logging.info("构建schema")
    if os.path.isfile(schema_path):
        logging.info("schema已存在")
        return len(json.load(open(schema_path,"r",encoding="utf8")))
    schema = set()
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            triplet = json.loads(line)
            attribute = triplet["attribute"]
            schema.add(attribute)
    schema.add("UNRELATED")
    logging.info("总共有%d个属性"%len(schema))
    output = {}
    writer = open(schema_path,"w",encoding="utf8")
    for index, attr in enumerate(schema):
        output[attr] = index
    writer.write(json.dumps(output,indent=2,ensure_ascii=False))
    writer.close()
    logging.info("schema构建完成")
    return len(output)

def load_vocab(vocab_path):
    output = {}
    with open(vocab_path,"r",encoding="utf8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            output[token] = idx + 1 # 0 for padding
    return output

class DataGenerator:
    def __init__(self,data_path,config,logger):
        '''
        data need to be : entity attribute value, and we treat value as a second entity
        return:
            input_id: sen_len, e1_mask:sen_len, e2_mask:sen_len, label: int
        '''
        self.config = config
        self.path = data_path
        self.logger = logger
        #use bert for tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_path"],use_fast=True)
        label_nums = build_schema(config["data_path"],config["schema_path"],logger)
        self.config["num_labels"] = label_nums
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.max_length = config["max_length"]
        self.load()
        self.logger.info("数据加载完成，共%d条数据，其中有%d条数据长度超过%d"%(len(self.data),self.exceed_max_length,self.max_length))
        self.logger.info("有%d条数据中的实体消失了"%(self.entity_disapper))

    def load(self):
        self.data = [] # 储存处理好的数据
        self.exceed_max_length = 0
        self.entity_disapper = 0

        with open(self.path,encoding="utf8") as f:
            for line in f:
                sample = json.loads(line)
                context = sample["context"]
                entity1 = sample["object"]
                attribute = sample["attribute"]
                entity2 = sample["value"]

                # process sentence
                try:
                    input_id, e1_mask, e2_mask, attr_label = self.process_sentence(context,entity1,attribute,entity2)
                except IndexError:
                    self.entity_disapper += 1
                    continue
                self.data.append([
                    torch.LongTensor(input_id),
                    torch.LongTensor(e1_mask),
                    torch.LongTensor(e2_mask),
                    torch.LongTensor([attr_label])
                ])

    def process_sentence(self,context,e1,attr,e2):
        '''
        Args:
            context: str
            e1: str
            attr: str
            e2: str
        return:
            input_id: sen_len, e1_mask:sen_len, e2_mask:sen_len, label: int
        '''
        # suppose context = "The kitchen is part of the house"
        # after preprocessing, context = "[CLS] The kitchen is part of the house"
        context= "[CLS]" + context
        if len(context) > self.max_length:
            self.exceed_max_length += 1
        encode_dict = self.tokenizer.encode_plus(
            context,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            return_offsets_mapping=True
        )
        input_id = encode_dict["input_ids"].squeeze(0) # sen_len
        #offset_mapping = [(0, 0), (0, 3), (4, 9), ...], ele[0] is the start position, ele[1] is the end position for each token
        offsets = encode_dict["offset_mapping"].squeeze(0) # sen_len, 2
        #locate entity position
        e1_mask = [0] * self.max_length
        e2_mask = [0] * self.max_length
        for idx, (start, end) in enumerate(offsets):
            if start == 0: continue
            if end == 0: break
            token = context[start:end]  # 举例：，[CLS, The, kitchen, is, part, of, the, house] idx:0, start-end:0-5 代表原来字符串里的索引，合起来是一个token

            # e1,e2 can be multi-token
            if token in e1:
                e1_mask[idx] = 1
            if token in e2:
                e2_mask[idx] = 1
        if not sum(e1_mask)>0 or not sum(e2_mask)>0:
            raise IndexError("lack of entity")

        #locate attribute mapping -> int
        attr_label = self.attribute_schema[attr]

        return input_id, e1_mask, e2_mask, attr_label

    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        '''
        return:
        list of 4 tensors
            input_id: sen_len, e1_mask:sen_len, e2_mask:sen_len, label: int
        '''
        return self.data[index]

def load_data(data_path, config,logger,shuffle=True):
    dg = DataGenerator(data_path,config,logger)
    dl = DataLoader(dg,config["batch_size"],shuffle=shuffle)
    return dl

if __name__ =='__main__':
    from config import Config
    dl = load_data(Config["data_path"],Config,logger)
    for batch in dl:
        input_id, e1_mask, e2_mask, label = batch
        print(input_id.shape)
        print(e1_mask.shape)
        print(e2_mask.shape)
        print(label.shape)
        break



