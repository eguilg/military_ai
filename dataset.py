import json
import logging
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from utils.read_elmo_embedding import get_elmo_vocab
from preprocess import preprocess_dataset
from vocab import Vocab

import multiprocessing


class MilitaryAiDataset(object):
  """
  This module implements the data loading and preprocessing steps
  """

  def __init__(self, train_preprocessed_files=[], train_raw_files=[],
               test_preprocessed_files=[], test_raw_files=[],
               char_embed_path="", token_embed_path="",
               elmo_vocab_path="", elmo_embed_path="",
               char_min_cnt=1, token_min_cnt=3,
               dev_split=0.1, seed=502, use_char_emb=False):
    self.logger = logging.getLogger("Military AI")

    self.train_set, self.test_set = [], []

    self.train_raw_path = train_raw_files
    self.test_raw_path = test_raw_files

    self.train_preprocessed_path = train_preprocessed_files
    self.test_preprocessed_path = test_preprocessed_files
    self.char_embed_path = char_embed_path
    self.token_embed_path = token_embed_path
    self.elmo_vocab_path = elmo_vocab_path
    self.elmo_embed_path = elmo_embed_path
    self.char_min_cnt = char_min_cnt
    self.token_min_cnt = token_min_cnt
    self.use_char_emb = use_char_emb
    self.all_tokens = []
    self.all_flags = []
    self.all_chars = []

    self._load_dataset()
    self._load_embeddings()
    # self._convert_to_ids()
    self._get_vocabs()
    self._convert_to_ids()

    self.p_max_tokens_len = max([sample['article_tokens_len'] for sample in self.train_set + self.test_set])
    self.q_max_tokens_len = max([sample['question_tokens_len'] for sample in self.train_set + self.test_set])

    self.p_token_max_len = max(
      [max([len(token) for token in sample['article_tokens']]) for sample in self.train_set + self.test_set])
    self.q_token_max_len = max(
      [max([len(token) for token in sample['question_tokens']]) for sample in self.train_set + self.test_set])
    #  split train & dev by article_id
    self.total_article_ids = sorted(list(set([sample['article_id'] for sample in self.train_set])))
    np.random.seed(seed)
    np.random.shuffle(self.total_article_ids)
    self.dev_article_ids = self.total_article_ids[:int(len(self.total_article_ids) * dev_split)]
    self.dev_set = list(filter(lambda sample: sample['article_id'] in self.dev_article_ids, self.train_set))
    self.train_set = list(filter(lambda sample: sample['article_id'] not in self.dev_article_ids, self.train_set))

  def _load_dataset(self):
    """
    Loads the dataset
    :return:
    """
    for i, train_file in enumerate(self.train_preprocessed_path):
      try:
        self.logger.info('Loading preprocessed train files...')
        trainSet = self._load_from_preprocessed(self.train_preprocessed_path[i])
      except FileNotFoundError:
        self.logger.info('Loading train files from raw...')
        trainSet = self._preprocess_raw(self.train_raw_path[i],
                                        self.train_preprocessed_path[i])
      self.train_set += trainSet
    for i, test_file in enumerate(self.test_preprocessed_path):
      try:
        self.logger.info('Try loading preprocessed test files...')
        testSet = self._load_from_preprocessed(self.test_preprocessed_path[i])
      except FileNotFoundError:
        self.logger.info('Loading train files from raw...')
        testSet = self._preprocess_raw(self.test_raw_path[i], self.test_preprocessed_path[i])
      self.test_set += testSet

  def _preprocess_raw(self, data_path, preprocessed_path):
    """
    Preprocess the raw data if preprocessed file doesn't exist
    :param data_path: the raw data path
    :param preprocessed_path:  the preprocessed file path to save
    :param train:  is training data
    :return:
        whole dataset
    """
    from feature_handler.question_handler import QuestionTypeHandler
    ques_type_handler = QuestionTypeHandler()

    articles, qas, dataset = preprocess_dataset(data_path)
    for article in articles:
        self.all_tokens.append(article['article_tokens'])
        self.all_chars.append(''.join(article['article_tokens']))
        self.all_flags.extend(article['article_flags'])
    for qa in qas:
        self.all_tokens.append(qa['question_tokens'])
        self.all_chars.append(''.join(qa['question_flags']))
        self.all_flags.extend(qa['question_flags'])
    for sample in dataset:
        question_types, type_vec = ques_type_handler.ana_type(''.join(sample['question_tokens']))
        sample['qtype'] = question_types
        sample['qtype_vec'] = type_vec.tolist()
        sample['article_tokens_len'] = len(sample['article_tokens'])
        sample['question_tokens_len'] = len(sample['question_tokens'])

    data_to_save = {
        'articles': articles,
        'questions': qas,
        'samples': dataset
    }

    with open(preprocessed_path, 'w') as fo:
        json.dump(data_to_save, fo)

    return dataset

  def _gen_hand_features(self, batch_data):
    batch_data['wiqB'] = []
    for sidx, sample in enumerate(batch_data['raw_data']):
      wiqB = [[0.0]] * batch_data['article_pad_len']
      for idx, token in enumerate(sample['article_tokens']):
        if token in sample['question_tokens']:
          wiqB[idx] = [1.0]
      batch_data['wiqB'].append(wiqB)
    return batch_data

  def _load_from_preprocessed(self, data_path):
    """
    Load preprocessed data if exists
    :param data_path: preprocessed data path
    :param train:  is training data
    :return: the whole dataset
    """
    from feature_handler.question_handler import QuestionTypeHandler
    ques_type_handler = QuestionTypeHandler()

    with open(data_path, 'r') as fp:
      total = json.load(fp)
      for article in total['articles']:
          self.all_tokens.append(article['article_tokens'])
          self.all_chars.append(''.join(article['article_tokens']))
          self.all_flags.extend(article['article_flags'])
      for qa in total['questions']:
          self.all_tokens.append(qa['question_tokens'])
          self.all_chars.append(''.join(qa['question_flags']))
          self.all_flags.extend(qa['question_flags'])
      for sample in total['samples']:
          question_types, type_vec = ques_type_handler.ana_type(''.join(sample['question_tokens']))
          sample['qtype'] = question_types
          sample['qtype_vec'] = type_vec.tolist()
          sample['article_tokens_len'] = len(sample['article_tokens'])
          sample['question_tokens_len'] = len(sample['question_tokens'])

    return total['samples']

  def _load_embeddings(self):

    try:
      self.logger.info("Loading char embedding model")
      self.char_wv = KeyedVectors.load(self.char_embed_path)
    except Exception:
      self.logger.info("Training char embedding model")
      self.char_wv = Word2Vec(self.all_chars, size=75, window=5, compute_loss=True,
                              min_count=1, iter=60, workers=multiprocessing.cpu_count()).wv
      self.logger.info("Saving char embedding model")
      self.char_wv.save(self.char_embed_path)
    try:
      self.logger.info("Loading token embedding model")
      self.token_wv = KeyedVectors.load(self.token_embed_path)
    except Exception:
      self.logger.info("Training token embedding model")
      self.token_wv = Word2Vec(self.all_tokens, size=300, window=5, compute_loss=True,
                               min_count=1, iter=60, workers=multiprocessing.cpu_count()).wv
      self.logger.info("Saving token embedding model")
      self.token_wv.save(self.token_embed_path)


    self.logger.info("Loading elmo embedding model")
    self.elmo_dict, self.elmo_embed = get_elmo_vocab(self.elmo_vocab_path, self.elmo_embed_path)

  def _get_vocabs(self):
    self.unique_flags = sorted(set(self.all_flags))

    def convert_to_one_hot(C):
      return np.eye(C)[np.linspace(0, C - 1, C - 1, dtype=np.int32).reshape(-1)].T

    flags_embeddings = convert_to_one_hot(len(self.unique_flags))

    self.unique_flags.insert(0, '<unk>')
    self.unique_flags.insert(0, '<pad>')
    flags_embeddings = np.concatenate([[np.zeros(flags_embeddings.shape[1], dtype=np.float32),
        np.ones(flags_embeddings.shape[1], dtype=np.float32) * 0.05, ], flags_embeddings], axis=0)

    self.token_wv.index2word.insert(0, '<unk>')
    self.token_wv.index2word.insert(0, '<pad>')
    self.token_wv.vectors = np.concatenate(
      [[np.zeros(self.token_wv.vector_size, dtype=np.float32),
        np.ones(self.token_wv.vector_size, dtype=np.float32) * 0.05, ], self.token_wv.vectors], axis=0)

    self.char_wv.index2word.insert(0, '<unk>')
    self.char_wv.index2word.insert(0, '<pad>')
    self.char_wv.vectors = np.concatenate(
      [[np.zeros(self.char_wv.vector_size, dtype=np.float32),
        np.ones(self.char_wv.vector_size, dtype=np.float32) * 0.05, ], self.char_wv.vectors], axis=0)

    self.flag_vocab = Vocab(self.unique_flags, flags_embeddings)

    self.char_vocab = Vocab(self.char_wv.index2word, self.char_wv.vectors)
    self.char_vocab.count([y for x in self.all_chars for y in x])
    unfiltered_char_vocab_size = self.char_vocab.size()
    self.char_vocab.filter_tokens_by_cnt(min_cnt=self.char_min_cnt)
    filtered_num = unfiltered_char_vocab_size - self.char_vocab.size()
    self.logger.info('After filter {} chars, the final char vocab size is {}'.format(filtered_num,
                                                                                     self.char_vocab.size()))

    self.token_vocab = Vocab(self.token_wv.index2word, self.token_wv.vectors)
    self.token_vocab.count([y for x in self.all_tokens for y in x])
    unfiltered_token_vocab_size = self.token_vocab.size()
    self.token_vocab.filter_tokens_by_cnt(min_cnt=self.token_min_cnt)
    filtered_num = unfiltered_token_vocab_size - self.token_vocab.size()
    self.logger.info('After filter {} tokens, the final token vocab size is {}'.format(filtered_num,
                                                                                       self.token_vocab.size()))
    self.elmo_vocab = Vocab(list(self.elmo_dict.keys()), self.elmo_embed)
    del self.elmo_dict, self.elmo_embed
    del self.char_wv, self.token_wv
    del self.all_chars, self.all_tokens, self.all_flags, self.unique_flags

  def _one_mini_batch(self, data, indices):
    """
    Get one mini batch
    Args:
        data: all data
        indices: the indices of the samples to be selected
        pad_id:
    Returns:
        one batch of data
    """
    batch_data = {'raw_data': [data[i] for i in indices],
                  'question_token_ids': [],
                  'article_token_ids': [],
                  'question_flag_ids': [],
                  'article_flag_ids': [],
                  'question_elmo_ids': [],
                  'article_elmo_ids': [],
                  'question_tokens_len': [],
                  'article_tokens_len': [],

                  'question_char_ids': [],
                  'article_char_ids': [],
                  'start_id': [],
                  'end_id': [],
                  'qtype_vecs': [],
                  'question_c_len': [],
                  'article_c_len': [],

                  # hand features
                  'wiqB': [],

                  'article_pad_len': 0,
                  'question_pad_len': 0,
                  'article_CL': 0,
                  'question_CL': 0,
                  }
    for sidx, sample in enumerate(batch_data['raw_data']):

      if self.use_char_emb:
        batch_data['question_char_ids'].append(sample['question_char_ids'])
        batch_data['article_char_ids'].append(sample['article_char_ids'])
      batch_data['question_token_ids'].append(sample['question_token_ids'])
      batch_data['question_tokens_len'].append(sample['question_tokens_len'])
      batch_data['article_token_ids'].append(sample['article_token_ids'])
      batch_data['article_tokens_len'].append(sample['article_tokens_len'])
      batch_data['question_flag_ids'].append(sample['question_flag_ids'])
      batch_data['article_flag_ids'].append(sample['article_flag_ids'])

      batch_data['question_elmo_ids'].append(sample['question_elmo_ids'])
      batch_data['article_elmo_ids'].append(sample['article_elmo_ids'])

      batch_data['qtype_vecs'].append(sample['qtype_vec'])
    batch_data, pad_p_len, pad_q_len, pad_p_token_len, pad_q_token_len = self._dynamic_padding(batch_data)
    batch_data['article_pad_len'] = pad_p_len
    batch_data['question_pad_len'] = pad_q_len
    if self.use_char_emb:
      batch_data['article_CL'] = pad_p_token_len
      batch_data['question_CL'] = pad_q_token_len
    for sample in batch_data['raw_data']:
      if 'answer_tokens' in sample and len(sample['answer_tokens']):

        batch_data['start_id'].append(sample['answer_token_start'])
        batch_data['end_id'].append(sample['answer_token_end'])
      else:
        # fake span for some samples, only valid for testing
        batch_data['start_id'].append(0)
        batch_data['end_id'].append(0)
    batch_data = self._gen_hand_features(batch_data)
    return batch_data

  def _dynamic_padding(self, batch_data):
    """
    Dynamically pads the batch_data with pad_id
    """

    pad_id_t = self.token_vocab.get_id(self.token_vocab.pad_token)
    pad_id_f = self.flag_vocab.get_id(self.flag_vocab.pad_token)
    pad_id_e = self.flag_vocab.get_id(self.elmo_vocab.pad_token)
    pad_id_c = self.char_vocab.get_id(self.char_vocab.pad_token)
    pad_p_len = min(self.p_max_tokens_len, max(batch_data['article_tokens_len']))
    pad_q_len = min(self.q_max_tokens_len, max(batch_data['question_tokens_len']))

    batch_data['article_token_ids'] = [(ids + [pad_id_t] * (pad_p_len - len(ids)))[: pad_p_len]
                                       for ids in batch_data['article_token_ids']]
    batch_data['question_token_ids'] = [(ids + [pad_id_t] * (pad_q_len - len(ids)))[: pad_q_len]
                                        for ids in batch_data['question_token_ids']]

    batch_data['article_flag_ids'] = [(ids + [pad_id_f] * (pad_p_len - len(ids)))[: pad_p_len]
                                       for ids in batch_data['article_flag_ids']]
    batch_data['question_flag_ids'] = [(ids + [pad_id_f] * (pad_q_len - len(ids)))[: pad_q_len]
                                        for ids in batch_data['question_flag_ids']]

    batch_data['article_elmo_ids'] = [(ids + [pad_id_e] * (pad_p_len - len(ids)))[: pad_p_len]
                                      for ids in batch_data['article_elmo_ids']]
    batch_data['question_elmo_ids'] = [(ids + [pad_id_e] * (pad_q_len - len(ids)))[: pad_q_len]
                                       for ids in batch_data['question_elmo_ids']]
    pad_p_token_len = self.p_token_max_len
    pad_q_token_len = self.q_token_max_len
    if self.use_char_emb:
      batch_data['article_c_len'] = []
      for article in batch_data['article_char_ids']:
        batch_data['article_c_len'] += [len(token) for token in article] + [0] * (pad_p_len - len(article))

      batch_data['question_c_len'] = []
      for question in batch_data['question_char_ids']:
        batch_data['question_c_len'] += [len(token) for token in question] + [0] * (pad_q_len - len(question))

      pad_p_token_len = min(self.p_token_max_len, max(batch_data['article_c_len']))
      pad_q_token_len = min(self.q_token_max_len, max(batch_data['question_c_len']))

      batch_data['article_char_ids'] = [
        ([(ids + [pad_id_c] * (pad_p_token_len - len(ids)))[:pad_p_token_len] for ids in tokens] + [
          [pad_id_c] * pad_p_token_len] * (pad_p_len - len(tokens)))[:pad_p_len] for tokens
        in batch_data['article_char_ids']]

      batch_data['question_char_ids'] = [
        ([(ids + [pad_id_c] * (pad_q_token_len - len(ids)))[:pad_q_token_len] for ids in tokens] + [
          [pad_id_c] * pad_q_token_len] * (pad_p_len - len(tokens)))[:pad_q_len] for tokens
        in batch_data['question_char_ids']]
    # print(len(batch_data))
    return batch_data, pad_p_len, pad_q_len, pad_p_token_len, pad_q_token_len

  def word_iter(self, set_name=None):
    """
    Iterates over all the words in the dataset
    Args:
        set_name: if it is set, then the specific set will be used
    Returns:
        a generator
    """
    if set_name is None:
      data_set = self.train_set + self.test_set
    elif set_name == 'train':
      data_set = self.train_set
    elif set_name == 'test':
      data_set = self.test_set
    else:
      raise NotImplementedError('No data set named as {}'.format(set_name))
    if data_set is not None:
      for sample in data_set:
        for token in sample['question_tokens']:
          yield token
        for token in sample['article_tokens']:
          yield token

  def _convert_to_ids(self):
    """
    Convert the question and article in the original dataset to ids
    Args:
        vocab: the vocabulary on this dataset
    """
    char2idx = self.char_vocab.token2id
    token2idx = self.token_vocab.token2id
    flag2idx = self.flag_vocab.token2id
    elmo2idx = self.elmo_vocab.token2id

    for data_set in [self.train_set, self.test_set]:
      if data_set is None:
        continue
      for sample in data_set:
        if self.use_char_emb:
          sample['question_char_ids'] = [
            [char2idx[c] if c in char2idx.keys() else char2idx['<unk>'] for c in token]
            for token in sample['question_tokens']]
          sample['article_char_ids'] = [
            [char2idx[c] if c in char2idx.keys() else char2idx['<unk>'] for c in token]
            for token in sample['article_tokens']]

        # sample['question_token_max_len'] = max([len(token) for token in sample['question_tokens']])
        # sample['article_token_max_len'] = max([len(token) for token in sample['article_tokens']])

        sample['question_token_ids'] = [token2idx[token] if token in token2idx.keys() else token2idx['<unk>']
                                        for token in sample['question_tokens']]
        sample['article_token_ids'] = [token2idx[token] if token in token2idx.keys() else token2idx['<unk>']
                                       for token in sample['article_tokens']]

        sample['question_flag_ids'] = [flag2idx[flag] if flag in flag2idx.keys() else flag2idx['<unk>']
                                       for flag in sample['question_flags']]
        sample['article_flag_ids'] = [flag2idx[flag] if flag in flag2idx.keys() else flag2idx['<unk>']
                                       for flag in sample['article_flags']]

        sample['question_elmo_ids'] = [elmo2idx[token] if token in elmo2idx.keys() else elmo2idx['<unk>']
                                       for token in sample['question_tokens']]
        sample['article_elmo_ids'] = [elmo2idx[token] if token in elmo2idx.keys() else elmo2idx['<unk>']
                                      for token in sample['article_tokens']]

  def gen_mini_batches(self, set_name, batch_size, shuffle=True):
    """
    Generate data batches for a specific dataset (train/dev/test)
    Args:
        set_name: train/dev/test to indicate the set
        batch_size: number of samples in one batch
        pad_id: pad id
        shuffle: if set to be true, the data is shuffled.
    Returns:
        a generator for all batches
    """
    if set_name == 'train':
      data = self.train_set
    elif set_name == 'dev':
      data = self.dev_set
    elif set_name == 'test':
      data = self.test_set
    else:
      raise NotImplementedError('No data set named as {}'.format(set_name))
    data_size = len(data)
    indices = np.arange(data_size)
    if shuffle:
      np.random.shuffle(indices)
    for batch_start in np.arange(0, data_size, batch_size):
      batch_indices = indices[batch_start: batch_start + batch_size]
      yield self._one_mini_batch(data, batch_indices)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)  # 设置日志级别
  data = MilitaryAiDataset(['./data/train/question_preprocessed.json'],
                           ['./data/train/question.json'],
                           char_embed_path='./data/embedding/char_embed75.wv',
                           token_embed_path='./data/embedding/token_embed300.wv',
                           elmo_vocab_path="./data/embedding/elmo-military_vocab.txt",
                           elmo_embed_path="./data/embedding/elmo-military_emb.pkl",)
