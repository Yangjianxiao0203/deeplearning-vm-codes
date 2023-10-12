import json
import math
import os

import torch.nn as nn
import torch
from RNNLanguageModel import build_model,build_vocab,build_dataset

window_size = None
char_dim = None
def load_model(model_path,param_path):
    # get json
    with open(param_path, 'r') as f:
        params = json.load(f)
    # get model.py
    vocab = build_vocab(params['vocab_path'])

    model = build_model(vocab, params['char_dim'])
    model.load_state_dict(torch.load(model_path))
    return model,params


def calc_perplexity(sentence, model):
    prob_total = 0
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [model.vocab.get(char, model.vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])

            if torch.cuda.is_available():
                x = x.cuda()

            logits = model(x)  # batch_size * len(vocab)
            probs = nn.functional.softmax(logits, dim=-1)

            target = sentence[i]
            target_index = model.vocab.get(target, model.vocab["<UNK>"])
            target_prob = probs[0][target_index]

            if target_prob.item() > 0:  # to avoid log(0)
                prob_total += math.log(target_prob.item(), 2)

    return 2 ** (-prob_total / len(sentence))


def load_models(model_dir):
    global window_size
    global char_dim
    model_paths = os.listdir(model_dir)
    class_to_model = {}
    for model_path in model_paths:
        if not model_path.endswith(".pth"):
            continue
        class_name = model_path.replace(".pth", "")
        model_path = os.path.join(model_dir, model_path)
        param_path = os.path.join(model_dir, class_name + ".json")
        model, params = load_model(model_path, param_path)
        # 更新全局变量的值
        window_size = params.get('window_size', window_size)
        char_dim = params.get('char_dim', char_dim)
        class_to_model[class_name] = model
    return class_to_model

def text_classification_based_on_language_model(class_to_model, sentence):
    ppl = []
    for class_name, class_lm in class_to_model.items():
        #用每个语言模型计算ppl
        ppl.append([class_name, calc_perplexity(sentence, class_lm)])
    ppl = sorted(ppl, key=lambda x:x[1])
    print(sentence)
    print(ppl[0: 3])
    print("==================")
    return ppl

if __name__ == "__main__":
    model_dir = "../data/corpus"
    models = load_models(model_dir)
    sentence = ["在全球货币体系出现危机的情况下",
                "点击进入双色球玩法经典选号图表",
                "慢时尚服饰最大的优点是独特",
                "做处女座朋友的人真的很难",
                "网戒中心要求家长全程陪护",
                "在欧巡赛扭转了自己此前不利的状态",
                "选择独立的别墅会比公寓更适合你",
                ]
    for s in sentence:
        text_classification_based_on_language_model(models, s)
