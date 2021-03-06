# import csv
#
#
# def _read_tsv(input_file, quotechar=None):
#     """Reads a tab separated value file."""
#     with open(input_file, "r", encoding="utf-8-sig") as f:
#         return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
#
#
# print(_read_tsv('data/test_debug.tsv'))

# from transformers import BertTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('chinese_wwm_pytorch')
# text = '宝宝爱宝贝~嘿嘿嘿'
# max_length = 15
# tokens = tokenizer.tokenize(text, add_special_tokens=True)
# print(tokens)
# tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
# print(tokens)
# text_len = len(tokens)
# input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))
# print(input_ids)
# # mask 0 表示mask掉
# attention_mask = [1] * text_len + [0] * (max_length - text_len)
# print(attention_mask)
# token_type_ids = [0] * max_length
# print(token_type_ids)

from transformers import BertForTokenClassification, BertConfig

config = BertConfig.from_pretrained(
        'chinese_wwm_pytorch',
        num_labels=17,
        finetuning_task='weiboner',
    )

model = BertForTokenClassification.from_pretrained(
        'chinese_wwm_pytorch',
        config=config,
    )

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

# print(model.modules)    # 查看模型整体结构，与model.children有区别
for layer in dict(model.named_parameters()).keys():
    print(layer)  # 查看网络层名称
# print(optimizer_grouped_parameters)

# import torch
# from torch.nn import Softmax
# t = torch.randn(32, 2)
# m = Softmax(dim=1)
# print(t)
# t = m(t)
# print(t)