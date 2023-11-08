import torch.cuda
import re
from loader import load_data

class Evaluator:
    def __init__(self,config,model,logger):
        self.config = config
        self.Debug = config["Debug"]
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_path"],config,logger,shuffle=False)
        self.bio_schema = self.valid_data.dataset.bio_schema # 用dataset可以获取原来dataGenerator的类
        self.attribute_schema = self.valid_data.dataset.attribute_schema #是个dict，key是str，value是int
        self.text_data = self.valid_data.dataset.text_data #每个元素都是个list，里面有四个str，分别代表是context, entity, attribute, value
        self.index_to_label = dict((y,x) for x,y in self.attribute_schema.items()) #是个dict，key是int，value是str

    @torch.no_grad()
    def eval(self,epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"object_acc":0,"attribute_acc":0,"value_acc":0,"full_match_acc":0}
        self.total = 0

        for batch_data,batch_text in self.valid_data:
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, attr_label, bio_label = batch_data
            #attr_pred: [batch_size,attribute_num]
            #bio_pred: [batch_size,seq_len,bio_num]
            attribute_pred, bio_pred = self.model(input_id)

            #compute running stats
            attr_pred = torch.argmax(attribute_pred,dim=-1,keepdim=True) # [batch_size,1]
            bio_pred = torch.argmax(bio_pred,dim=-1) # [batch_size,seq_len]

            # for attr_p,bio_p,info in zip(attr_pred,bio_pred,batch_text):
            #     context, entity, attribute, value = info
            for idx, (attr_p,bio_p) in enumerate(zip(attr_pred,bio_pred)):
                context, entity,attribute,value = batch_text[0][idx],batch_text[1][idx],batch_text[2][idx],batch_text[3][idx]
                #docode attr_p and bio_p to compare with ground truth
                attr_p = attr_p.cpu().detach().tolist()[0] #int
                bio_p = bio_p.cpu().detach().tolist() # (seq_len, )
                entity_p, value_p = self.decode(bio_p,context)

                self.stats_dict["object_acc"] += int(entity_p == entity)
                self.stats_dict["attribute_acc"] += int(attr_p == self.attribute_schema[attribute])
                self.stats_dict["value_acc"] += int(value_p == value)
                self.stats_dict["full_match_acc"] += int(entity_p == entity and attr_p == self.attribute_schema[attribute] and value_p == value)
            self.total += len(batch_text)
            if self.Debug:
                break
        self.show_statis()
        self.logger.info("第%d轮模型测试结束" % epoch)
        return


    def decode(self,bio_pred,context):
        '''
        bio_pred: [seq_len]
        '''
        bio_pred_str = "".join(str(i) for i in bio_pred)
        entity_pred = self.seek_pattern("01*",bio_pred_str,context)
        value_pred = self.seek_pattern("23*",bio_pred_str,context)
        return entity_pred,value_pred

    def seek_pattern(self,pattern, pred_label,context):
        '''
        pattern: regulation expression
        pred_label: [seq_len] for context, with padding
        context: str
        '''
        pred_obj =""
        match = re.search(pattern,context) # return match object
        if match:
            start,end = match.span() # return start and end index
            pred_obj = pred_label[start:end]
        return pred_obj


    def show_statis(self):
        for key,value in self.stats_dict.items():
            self.logger.info("%s: %d" % (key,value/self.total))

        self.logger.info("*"*20)
        return