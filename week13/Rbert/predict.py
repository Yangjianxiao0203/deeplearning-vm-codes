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


if __name__=='__main__':
    model_path = "model_output/epoch_15.pth"
    pred = Predict(Config,model_path,logger)