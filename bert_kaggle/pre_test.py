import torch
from transformers import *

if torch.cuda.is_available():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    device = torch.device("cuda")
    print("current GPU :", torch.cuda.current_device())
    print('GPU:',torch.cuda.get_device_name)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import os
import glob

# from transformers import  WEIGHTS_NAME
# print(WEIGHTS_NAME)
# ch = list(os.path.dirname(c) for c in sorted(glob.glob("tnews" + "/**/" + WEIGHTS_NAME, recursive=True)))
# print(ch)

# pretrained_weights = 'chinese_wwm_pytorch'
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# model = BertForTokenClassification.from_pretrained(pretrained_weights, num_labels=4)
# input_ids = torch.tensor(tokenizer.encode("夕小瑶的卖萌屋")).unsqueeze(0) 
# labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)
# print(input_ids.size)
# print(input_ids.size(1))
# 
# print(labels)
# print(input_ids)
# outputs = model(input_ids, labels=labels)
# loss, scores = outputs[:2]
# print(loss, scores)
# 
# # tokenizer = BertTokenizer.from_pretrained("bert-base_chinese")
# # model = BertModel.from_pretrained("bert-base_chinese")
# # 
# # # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
# # 
# input_ids = torch.tensor([tokenizer.encode("夕小瑶的卖萌屋", add_special_tokens=True)])  
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]
#     print(last_hidden_states)
# 


