from transformers import AutoTokenizer


import torch
def process_sentence(context, e1, attr, e2):
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
    context = "[CLS]" + context
    # e1_marked = f" $ {e1} $ "
    # e2_marked = f" # {e2} # "
    # context = context.replace(e1,e1_marked)
    # context = context.replace(e2,e2_marked)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
    encode_dict = tokenizer.encode_plus(
        context,
        truncation=True,
        max_length=10,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    input_id = encode_dict["input_ids"].squeeze(0)  # sen_len
    # offset_mapping = [(0, 0), (0, 3), (4, 9), ...], ele[0] is the start position, ele[1] is the end position for each token
    offsets = encode_dict["offset_mapping"].squeeze(0)
    # locate entity position
    e1_mask = [0] * 10
    e2_mask = [0] * 10
    for idx, (start, end) in enumerate(offsets):
        if start == 0: continue
        if end == 0: break
        token = context[start:end]
        if token in e1:
            e1_mask[idx] = 1
        if token in e2:
            e2_mask[idx] = 1
    return input_id, e1_mask, e2_mask

if __name__ =='__main__':
    context = "The kitchen is part of the house"
    e1 = "kitchen"
    attr = "part of"
    e2 = "house"
    input_id, e1_mask,e2_mask = process_sentence(context, e1, attr, e2)
    print(input_id)
    print(e1_mask)
    print(e2_mask)