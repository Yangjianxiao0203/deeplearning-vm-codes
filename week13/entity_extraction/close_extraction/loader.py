import torch
import json
import os
from config import Config
import logging
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    '''
    load the vocab file
    '''
    token_dict = {}
    with open(vocab_path,"r",encoding="utf8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            token_dict[token] = idx + 1  # 0 for padding
    return token_dict

class DataGenerator:
    def __init__(self,data_path, config,logger):
        '''
        序列标注：BIO B表示开始，I表示中间，O表示结束， B-object， I-object， B-value， I-value，O
        '''
        self.config = config
        self.data_path = data_path
        self.logger = logger

        attr_count = build_schema(config["data_path"],config["schema_path"],logger)
        config["attribute_count"] = attr_count # set config attr_count

        self.attribute_schema = json.load(open(config["schema_path"],"r",encoding="utf8"))
        self.vocab = load_vocab(config["vocab_path"])
        config["vocab_size"] = len(self.vocab)  # set config vocab_size

        self.max_length = config["max_length"]
        self.exceed_max_length = 0
        #BIO:只要实体和属性值，属性直接归类到O中
        self.bio_schema = {
            "B_object": 0,
            "I_object": 1,
            "B_value": 2,
            "I_value": 3,
            "O": 4,
            "PAD": -100
        }
        config["bio_count"] = len(self.bio_schema) # set config bio_count
        self.load()
        self.logger.info("超出设定最大长度的样本数量:%d, 占比:%.3f" % (
            self.exceed_max_length, self.exceed_max_length / len(self.data)))
        self.logger.info("数据加载完成,总共有%d个样本" % len(self.data))
        return

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        '''
        return:
        self.data[index]: list of 3, 代表处理好的真实文本
            0: input_id: 1 * max_length
            1: attribute_label: int
            2: sentence_label: 1 * max_length
        self.text_data[index]: list of 4, 代表真实的文本
            0: context: str
            1: entity: str
            2: attribute: str
            3: value: str
        '''
        return self.data[index],self.text_data[index]

    def load(self):
        path = self.data_path
        self.data = [] #每个元素都是个list，里面有三个torch Long Tensor，分别代表是input_id, attribute_label, sentence_label
        self.text_data = [] #每个元素都是个list，里面有四个str，分别代表是context, entity, attribute, value

        with open(path,"r",encoding="utf8") as f:
            for line in f:
                triplet = json.loads(line)
                context = triplet["context"]
                entity = triplet["object"]
                attribute = triplet["attribute"]
                value = triplet["value"]
                if attribute not in self.attribute_schema:
                    attribute = "UNRELATED"
                self.text_data.append([context, entity,attribute,value])
                input_id, attribute_label, sentence_label = self.process_sentence(context, entity, attribute, value)
                self.data.append([
                    torch.LongTensor(input_id),
                    torch.LongTensor([attribute_label]),
                    torch.LongTensor(sentence_label)
                ])
        return

    def process_sentence(self, context, entity, attribute, value):
        '''
        Args:
            context: str
            entity: str
            attribute: str
            value: str
        Returns:
            input_id: list represent the sentence encoding
            attribute_label: it's a schema index, int
            sentence_label: list represent the sentence label, represents the bio label for each word, the same length as input_id
        流程：
            1. 找到实体和属性值在句子中的位置
            2. 找到属性的schema index
            3. 对句子进行编码，得到input_id，input id必须和句子长度一致
            4. 对句子进行BIO标注，得到sentence_label，sentence_label必须和句子长度一致
            5. 对input_id和sentence_label进行padding，使得长度和max_length一致
        '''
        if len(context) > self.max_length:
            self.exceed_max_length += 1
        # find the start of entity
        entity_start = context.find(entity)
        entity_end = entity_start + len(entity)
        # find the start of value
        value_start = context.find(value)
        value_end = value_start + len(value)
        # find the index of attribute
        attribute_label = self.attribute_schema[attribute]
        # encoding the sentence
        input_id = self.encode_sentence(context)
        assert len(input_id) == len(context), "input_id长度不一致"

        # BIO
        sentence_label = [self.bio_schema["O"]] * len(input_id)
        sentence_label[entity_start] = self.bio_schema["B_object"]
        for i in range(entity_start+1,entity_end):
            sentence_label[i] = self.bio_schema["I_object"]
        sentence_label[value_start] = self.bio_schema["B_value"]
        for i in range(value_start+1,value_end):
            sentence_label[i] = self.bio_schema["I_value"]

        #padding
        input_id = self.padding(input_id)
        sentence_label = self.padding(sentence_label,padding_token=self.bio_schema["PAD"])

        return input_id, attribute_label, sentence_label

    def encode_sentence(self,text,padding=False):
        input_id = []
        for token in text:
            input_id.append(self.vocab.get(token,self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self,input_id, padding_token = 0):
        '''
        cut the sentence if it's too long
        pad the sentence if it's too short, pad in the end
        '''
        input_id = input_id[:self.max_length]
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [padding_token] * pad_len
        return input_id

def load_data(data_path, config,logger,shuffle=True):
    dg = DataGenerator(data_path,config,logger)
    dl = DataLoader(dg,config["batch_size"],shuffle=shuffle)

    return dl

if __name__ =='__main__':
    train_path = Config["train_path"]
    data_loader = load_data(train_path,Config,logger)
    for idx, data in enumerate(data_loader):
        batch, batch_text = data
        input_id, attribute_label, sentence_label = batch
        context, entity, attribute, value = batch_text
        print("input_id shape: ",input_id.shape)
        print("attribute_label shape: ",attribute_label.shape)
        print("sentence_label shape: ",sentence_label.shape)
        print("context shape: ",len(context))
        print("entity : ",entity)
        print("attribute: ",attribute)
        print("value: ",value)
        break