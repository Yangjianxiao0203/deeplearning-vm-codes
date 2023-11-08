import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from sklearn.metrics import classification_report

class Evaluator:
    def __init__(self,config,model,logger):
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_path"],config,logger,shuffle=False)
        self.attribute_schema = self.valid_data.dataset.attribute_schema
        self.index_to_label = dict((y,x) for x,y in self.attribute_schema.items())

    @torch.no_grad()
    def eval(self,epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        gold = []
        pred = []

        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, e1_mask, e2_mask, labels = batch_data
            gold.append(labels.detach().tolist())
            attr_pred = self.model(input_id,e1_mask,e2_mask)
            attr_pred = torch.argmax(attr_pred,dim=-1)
            pred.append(attr_pred.detach().tolist())
        report = classification_report(np.array(gold), np.array(pred)).rstrip().split("\n")
        self.logger.info(report[0])
        self.logger.info(report[-1])

        return