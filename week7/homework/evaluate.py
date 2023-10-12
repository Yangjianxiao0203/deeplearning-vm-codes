import torch
import torch.nn as nn

from week7.homework.loader import load_data


class Evaluator:
    def __init__(self,config,model,logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["data_path"],config,shuffle=False)
        self.correct = 0
        self.total = 0

    def eval(self,epoch):
        self.model.eval()
        self.correct = 0
        self.total = 0
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        for index,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            x,y = batch_data
            with torch.no_grad():
                y_pred = self.model(x) # batch * class_num
            #choose max in y_pred
            y_pred = torch.argmax(y_pred,dim=-1)  # batch * 1
            #compare y_pred and y
            correct = (y_pred == y).sum().item()
            self.correct += correct
            self.total += y.shape[0]
        acc = self.correct / self.total
        self.logger.info("预测准确率：%f" % acc)


