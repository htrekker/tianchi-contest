import argparse
# import torch
import csv
import json
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser('Input those hyper-parameters.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epoch', type=int)

    return parser.parse_args()


def get_label_distribution():
    postive_cnt, negative_cnt = 0, 0

    with open('data/shuffled_dataset.tsv', mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[2] == '1':
                postive_cnt += 1
            else:
                negative_cnt += 1

    print('postive %d, negative %d' % (postive_cnt, negative_cnt))


def do():
    postive_cnt, negative_cnt = 0, 0
    with open('result.txt', mode='r') as f:
        for row_num, row in enumerate(f):
            result = float(row)
            if result > 0.5 and result - 0.5 > 0.4:
                postive_cnt += 1
                print('row: %d: %.5f' % (row_num, result))
            elif result < 0.5 and 0.5 - result > 0.4:
                negative_cnt += 1
                print('row: %d: %.5f' % (row_num, result))

    print('postive: %d, negative: %d.' % (postive_cnt, negative_cnt))


def get_glove_token_dict():
    glove_token_dict = {}

    with open('glove_vectors.txt', mode='r') as f:
        for row_num, row in enumerate(f):
            token = row.split()[0]
            glove_token_dict[token] = row_num + 1
    glove_token_dict['0'] = 0

    with open('golve_token2id.json', mode='w') as f:
        json.dump(glove_token_dict, f)

    # pad = np.zeros(50)


get_glove_token_dict()
