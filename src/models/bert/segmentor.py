import torch


def seg(text, model, tokenizer, cvt):
    input_ids = []
    i = 0
    while i < len(text):
        input_ids += tokenizer.encode(text[i:(i+100)], add_special_tokens=False)
        i += 100
    length = len(input_ids)
    input_ids.insert(0, 1) # id of [CLS]
    input_ids.insert(length + 1, 2) # id of [SEP]
    vecs = (model(torch.tensor(input_ids).unsqueeze(0), mode=1))[0][0]
    labels = []
    labels.append(0)
    for i in range(2, length + 1):
        if (vecs[i][0] > vecs[i][1]):
            labels.append(0)
        else:
            labels.append(1)
    result = cvt.label2text(text, labels)
    return result
