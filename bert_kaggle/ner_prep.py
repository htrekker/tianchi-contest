#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import csv
import pandas as pd
import sys

ner_labels_dict = []
data = []
sentence = []
label_list = []
assert len(sys.argv) == 3, "usage: python ner_prep.py <原始文件路径> <保存路径> "

file_path = sys.argv[1]
save_name = sys.argv[2]


with codecs.open(file_path, "r", "utf-8") as f:
    for line in f:
        if line.strip() != "":
            tmp= line.strip().split("\t")
            if tmp[-1] not in ner_labels_dict:
                ner_labels_dict.append(tmp[-1])
            sentence.append(tmp[0][0])
            label_list.append(tmp[-1])
        elif line == "\n":
            data.append((sentence, label_list))
            sentence = []
            label_list = []
        else:
            continue

with open("%s.tsv" % name, "w") as f:
    for ex in data:
        s = "".join(ex[0])
        l = " ".join(ex[1])
        f.write(s+"\t"+l+"\n")

if save_name.find("train.tsv") > -1:
    with open("label.txt", "w") as f:
        for k in ner_labels_dict:
            f.write(k+"\n")
