# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
from enum import Enum
from typing import List, Optional, Union
import dataclasses
import numpy as np
import json
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from src.transformers.data.processors import DataProcessor
from src.transformers.file_utils import is_tf_available
from src.transformers.tokenization_utils import PreTrainedTokenizer

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class InputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

# 数据类，相比于普通类更加易读，实现了一些简单的方法，eg. > == 等比较运算符
@dataclass
class NERInputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)  # frozen=True 对字段赋值将会产生异常, 模拟了只读的冻结实例
class InputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[Union[int, float]]
    tokens2: Optional[List[str]] = None
    input_ids2: Optional[List[int]] = None
    attention_mask2: Optional[List[int]] = None
    token_type_ids2: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class NERInputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[List[Union[int, float]]]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class ChnSentiProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class LCQMCProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))     # matching
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]    # str '1 2 3'
            text_b = line[1]    # str '4 5 6'
            label = line[2]     # str '0'
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[0]
            text_b = line[1]
            label = '0'     # kaggle 设置为0 但不用
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))     # label = None
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class WeiboNerProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        # _read_tsv：返回一个列表，[[sen, label],[sen, label],...,[sen, label]]
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))     # ner
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)   # train-i
            # ner只有text_a
            text_a = line[0]    # sentence
            label = line[1].split(" ")  # label eg. "B I O O O O" .split(" ") 成为列表 [B,I,O,O,O,O]
            examples.append(NERInputExample(guid=guid, text_a=text_a,  label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[0]
            label = line[1].split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels   # 返回标签列表


processors = {
    "lcqmc": LCQMCProcessor,
    "chnsenti": ChnSentiProcessor,
    "weiboner": WeiboNerProcessor
    }


output_modes = {
    "chnsenti": "classification",
    "lcqmc": "classification",
    "weiboner": "ner"
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):                                                                           
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            

def convert_examples_to_features(
        # 联合类型；Union[X, Y] 的意思是，非 X 即 Y
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        # Optional[X] 等价于 Union[X, None]
        max_length: Optional[int],
        label_list: List[str],
        output_mode: str
    ):
    # for ner task
    def convert_text_to_ids(text):

        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        tokens = ["[CLS]"]+tokens[:max_length-2]+["[SEP]"]
        text_len = len(tokens)
        # [PAD] 的id是0
        input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
        # mask 0 表示mask掉
        attention_mask = [1]*text_len+[0]*(max_length-text_len)
        # ner中相同，均为0
        token_type_ids = [0]*max_length

        # 检查长度是否正确
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids

    def convert_text_to_ids_for_matching(text_a, text_b):
        max_num = 21128
        tokens_a = [int(num) for num in text_a.split()]
        tokens_b = [int(num) for num in text_b.split()]
        for i in range(len(tokens_a)):
            if tokens_a[i] > max_num:   tokens_a[i] = max_num - 10
        for i in range(len(tokens_b)):
            if tokens_b[i] > max_num:   tokens_b[i] = max_num - 10

        if len(tokens_a) + len(tokens_b) > (max_length-3):
            _truncate_seq_pair(tokens_a, tokens_b, max_length-3)
        # [CLS] 101 [SEP] 102 [PAD] 0
        tokens = [101] + tokens_a + [102] + tokens_b + [102]
        text_len = len(tokens)
        input_ids = tokens+[0]*(max_length-text_len)
        attention_mask = [1]*text_len+[0]*(max_length-text_len)
        token_type_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b)+1)+[0]*(max_length-text_len)
        
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for i in range(0, len(examples)):

        if examples[i].text_b:
            tokens1, input_ids1, attention_mask1, token_type_ids1 = convert_text_to_ids_for_matching(examples[i].text_a, examples[i].text_b)
        else:
            # ner task
            tokens1, input_ids1, attention_mask1, token_type_ids1 = convert_text_to_ids(examples[i].text_a)
        
        if output_mode == "ner":
            label_id = [label_map["O"]]     # tokens中第一个字符为'[CLS]' 标记为O
            for j in range(len(tokens1)-2):
                label_id.append(label_map[examples[i].label[j]])   # examples[i].label[j]为标记列表 eg.[B,I,O,O,O,O]
            label_id.append(label_map["O"])     # tokens中最后一个字符为'[SEP]' 标记为O
            if len(label_id) < max_length:
                label_id = label_id + [label_map["O"]]*(max_length-len(label_id))    # 将剩余的[PAD]标记为 O
        # matching
        else:
            label_id = label_map[examples[i].label]

        feature = InputFeatures(
            tokens1=tokens1,
            input_ids1=input_ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            label_id=label_id)

        features.append(feature)

    return features


def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]    # 'ner'
    logger.info("Creating features from dataset file at %s", args.data_dir)
    # 标签种类列表 eg. [B, I, O]
    label_list = processor.get_labels(args.data_dir)
    if evaluate:
        examples = (
            processor.get_test_examples(args.data_dir, args.input_test_name)
        )
    else:   # train
        examples = (
            processor.get_train_examples(args.data_dir)
        )
    features = convert_examples_to_features(
        # default args.max_seq_length = 128
        examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
    )

    # Convert to Tensors and build dataset
    # 装换成tensor
    all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)   # torch.long 64bit
    all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # tensordataset 类似 zip 打包
    dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1,  all_labels)

    return dataset, examples

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


def compute_metrics(preds, labels):
    return {"acc": (preds == labels).mean()}


def ner_F1(preds, labels, mask_indicators):
    assert len(preds) == len(labels) == len(mask_indicators)
    print(preds.shape)
    print(labels.shape)
    print(preds[0])
    print(labels[0])
    total_preds = []
    total_ground = []
    for i in range(len(preds)):
        num = sum(mask_indicators[i]) - 2
        total_preds.extend(preds[i][1: 1+num])
        total_ground.extend(labels[i][1: 1+num])

    refer_label = total_ground
    pred_label = total_preds
    fn = dict()
    tp = dict()
    fp = dict()
    for i in range(len(refer_label)):
        if refer_label[i] == pred_label[i]:
            if refer_label[i] not in tp:
                tp[refer_label[i]] = 0
            tp[refer_label[i]] += 1
        else:
            if pred_label[i] not in fp:
                fp[pred_label[i]] = 0
            fp[pred_label[i]] += 1
            if refer_label[i] not in fn:
                fn[refer_label[i]] = 0
            fn[refer_label[i]] += 1
    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())
    p_total = float(tp_total) / (tp_total + fp_total)
    r_total = float(tp_total) / (tp_total + fn_total)
    f_micro = 2 * p_total * r_total / (p_total + r_total)
    
    return {"f1_score": f_micro}
        

if _has_sklearn:
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

