import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
import pandas as pd

import json

df = pd.read_csv('data/gaiic_track3_round1_train_20210228.tsv',
                 header=None, sep='\t')
df2 = pd.read_csv('data/gaiic_track3_round1_testA_20210228.tsv',
                  header=None, sep='\t')
print(df.head())

sentences = df[0].to_list()
sentences.extend(df[1].to_list())
sentences.extend(df2[0].to_list())
sentences.extend(df2[1].to_list())

sentences = list(map(lambda x: x.split(), sentences))

print(len(sentences))

emb_size = 50

model = word2vec.Word2Vec(sentences, min_count=1, size=emb_size, iter=20)
# print(model.wv)


model.wv.save_word2vec_format('word_embeddings.kv')

embeddings = KeyedVectors.load_word2vec_format('word_embeddings.kv')

embeddings.add('0', np.zeros(emb_size))

embeddings.save_word2vec_format('word_embeddings.kv')
print(embeddings.wv.vocab['0'])
print(model.wv.similarity('12', '26'))

token_list = embeddings.wv.index2word

token2id = {}

for i, token in enumerate(token_list):
    if i < 10:
        print(token, i+1)
    token2id[token] = i+1

with open('token2id.json', mode='w') as f:
    json.dump(token2id, fp=f)

# model.save('word_embeddings')
