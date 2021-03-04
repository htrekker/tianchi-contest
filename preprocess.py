from gensim.models import word2vec
import pandas as pd

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

model = word2vec.Word2Vec(sentences, min_count=1, size=emb_size)
# print(model.wv)

print(len(model.wv.vocab))
# print(model.wv.vocab['0'])
print(model.wv.similarity('12', '26'))

model.save('word_embeddings')
