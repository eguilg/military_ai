# coding = utf-8
import pickle
import os
import numpy as np


def pickle_dump(data, file_path):
  f_write = open(file_path, 'wb')
  pickle.dump(data, f_write, True)


def pickle_load(file_path):
  f_read = open(file_path, 'rb')
  data = pickle.load(f_read)

  return data


# data_folder = '../data/embedding'
# vocab_path = os.path.join(data_folder, 'elmo-military_vocab.txt')
# embedding_path = os.path.join(data_folder, 'elmo-military_emb.pkl')


def get_elmo_vocab(vocab_path, embedding_path):
  word_to_id = {}
  word_to_id['<pad>'] = 0
  with open(vocab_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[2:]:  ## 这里的[2:]是去掉开头的<S>和</S>
      word_to_id[line.strip()] = len(word_to_id)

  embeddings = pickle_load(embedding_path)
  embeddings = embeddings[2:]
  embeddings = np.concatenate([[np.zeros(len(embeddings[0]), dtype=np.float32)], embeddings], axis=0)
  assert len(word_to_id) == len(embeddings)

  return word_to_id, embeddings


# word_to_id, embeddings = get_elmo_vocab(vocab_path, embedding_path)
# pad_id = word_to_id['<pad>']
# assert pad_id == 0
# unk_id = word_to_id['<unk>']
# assert unk_id == 1
#
# tokens = '2015 年 正向 我们 走来 ， 新年 的 钟声 就要 敲响 了'.split()
# print(embeddings.shape)
# for token in tokens:
#   print(token, word_to_id[token], len(embeddings[word_to_id[token]]) if token in word_to_id else unk_id)
