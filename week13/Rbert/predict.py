from model import Rbert
from config import Config
from transformers import AutoTokenizer
import json
import torch
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predict:
    def __init__(self,config,model_path,logger):
        self.config = config
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())
        config["num_labels"] = len(self.attribute_schema)
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_path"],use_fast=True)
        self.model = Rbert(config,logger)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()
        logger.info("模型加载完毕!")

    @torch.no_grad()
    def predict(self,sentence,e1,e2):
        input_id, e1_mask, e2_mask = self._process_sentence(sentence,e1,e2)
        attr_pred = self.model(input_id,e1_mask,e2_mask) # 1* num_labels
        attr_pred = torch.argmax(attr_pred,dim=-1).item() # 1
        attr_pred = self.index_to_label[attr_pred]
        return attr_pred

    def _process_sentence(self,context,e1,e2):
        '''
        context : seq_len
        '''
        x_tokenized = self.tokenizer.encode_plus(
            text=context,
            add_special_tokens=False,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_id = x_tokenized["input_ids"].squeeze(0) # (seq_len,)
        token_dict = x_tokenized["offset_mapping"].squeeze(0) # seq_len*2
        e1_mask = torch.zeros_like(input_id)
        e2_mask = torch.zeros_like(input_id)

        for idx,(start,end) in enumerate(token_dict):
            if start == 0:
                continue
            if end == 0:
                break
            token = context[start:end]
            if token in e1:
                e1_mask[idx] = 1
            if token in e2:
                e2_mask[idx] = 1
        input_id = input_id.unsqueeze(0) # 1*seq_len
        e1_mask = e1_mask.unsqueeze(0) # 1*seq_len
        e2_mask = e2_mask.unsqueeze(0) # 1*seq_len
        return input_id,e1_mask,e2_mask


if __name__=='__main__':
    model_path = "model_output/epoch_15.pth"
    sl = Predict(Config,model_path,logger)

    sentence = "可你知道吗，兰博基尼的命名取自创始人“费鲁吉欧·兰博基尼”的名字，而更让人意外的是，兰博基尼刚开始只是一个做拖拉机的！"
    e1 = "兰博基尼"
    e2 = "费鲁吉欧·兰博基尼"
    res = sl.predict(sentence, e1, e2)
    print("预测关系：", res)

    sentence = "傻丫头郭芙蓉、大女人翠平、励志的杜拉拉，姚晨的角色跳跃很大，是一个颇能适应各种类型题材的职业演员。"
    e1 = "姚晨"
    e2 = "演员"
    res = sl.predict(sentence, e1, e2)
    print("预测关系：", res)