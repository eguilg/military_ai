# coding = utf-8
import gc
import json
import logging
import multiprocessing

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from utils.read_elmo_embedding import get_elmo_vocab
from vocab import Vocab


class MilitaryAiDataset(object):
	"""
	This module implements the data loading and preprocessing steps
	"""

	def __init__(self, pyltp_cfg, jieba_cfg, use_jieba=False, test=False):
		self.logger = logging.getLogger("Military AI")
		self.logger.setLevel(logging.INFO)
		self.train_set, self.test_set = [], []

		self.train_raw_path = pyltp_cfg.train_raw_file
		self.test_raw_path = pyltp_cfg.test_raw_file
		self.use_jieba = use_jieba
		self.pyltp_cfg = pyltp_cfg
		self.jieba_cfg = jieba_cfg
		self.pyltp_train_preprocessed_path = pyltp_cfg.train_preprocessed_file
		self.pyltp_test_preprocessed_path = pyltp_cfg.test_preprocessed_file
		self.pyltp_flag_embed_path = pyltp_cfg.flag_embed_file
		self.pyltp_token_embed_path = pyltp_cfg.token_embed_file

		self.jieba_train_preprocessed_path = jieba_cfg.train_preprocessed_file
		self.jieba_test_preprocessed_path = jieba_cfg.test_preprocessed_file
		self.jieba_flag_embed_path = jieba_cfg.flag_embed_file
		self.jieba_token_embed_path = jieba_cfg.token_embed_file

		self.elmo_vocab_path = pyltp_cfg.elmo_dict_file
		self.elmo_embed_path = pyltp_cfg.elmo_embed_file
		self.seed = pyltp_cfg.seed
		self.dev_split = pyltp_cfg.dev_split
		self.cv = pyltp_cfg.cv

		self.is_test = test

		if use_jieba:
			self.train_preprocessed_path = self.jieba_train_preprocessed_path
			self.test_preprocessed_path = self.jieba_test_preprocessed_path
		else:
			self.train_preprocessed_path = self.pyltp_train_preprocessed_path
			self.test_preprocessed_path = self.pyltp_test_preprocessed_path

		self._load_dataset()
		self._load_embeddings()
		# self._convert_to_ids()
		self._get_vocabs()

		if use_jieba:
			self.flag_vocab = self.jieba_flag_vocab
			self.token_vocab = self.jieba_token_vocab
		else:
			self.flag_vocab = self.pyltp_flag_vocab
			self.token_vocab = self.pyltp_token_vocab

		self._convert_to_ids()

		self.p_max_tokens_len = max([sample['article_tokens_len'] for sample in self.train_set + self.test_set])
		self.q_max_tokens_len = max([sample['question_tokens_len'] for sample in self.train_set + self.test_set])

		self.p_token_max_len = max(
			[max([len(token) for token in sample['article_tokens']]) for sample in self.train_set + self.test_set])
		self.q_token_max_len = max(
			[max([len(token) for token in sample['question_tokens']]) for sample in self.train_set + self.test_set])

		if not self.is_test:
			#  split train & dev by article_id
			self.total_article_ids = sorted(list(set([sample['article_id'] for sample in self.train_set])))
			np.random.seed(self.seed)
			np.random.shuffle(self.total_article_ids)
			one_piece = int(len(self.total_article_ids) * self.dev_split)
			self.dev_article_ids = self.total_article_ids[(self.cv - 1) * one_piece: self.cv * one_piece]
			self.dev_article_ids = self.total_article_ids[:int(len(self.total_article_ids) * self.dev_split)]
			self.dev_set = list(
				filter(lambda sample: sample['article_id'] in self.dev_article_ids and sample['answer_token_start'] >= 0,
					   self.train_set))
			self.train_set = list(filter(
				lambda sample: sample['article_id'] not in self.dev_article_ids and sample['answer_token_start'] >= 0,
				self.train_set))

	#
	def _load_dataset(self):
		"""
		Loads the dataset
		:return:
		"""
		if not self.is_test:
			try:
				self.logger.info('Loading preprocessed train files...')
				self.train_set = self._load_from_preprocessed(self.train_preprocessed_path)
			except FileNotFoundError:
				self.logger.info('Preprocessed train file not found !')
		else:
			try:
				self.logger.info('Try loading preprocessed test files...')
				self.test_set = self._load_from_preprocessed(self.test_preprocessed_path)
			except FileNotFoundError:
				self.logger.info('Preprocessed test file not found !')



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

			for sample in total:
				question_types, type_vec = ques_type_handler.ana_type(''.join(sample['question_tokens']))
				sample['qtype'] = question_types
				sample['qtype_vec'] = type_vec.tolist()
				sample['article_tokens_len'] = len(sample['article_tokens'])
				sample['question_tokens_len'] = len(sample['question_tokens'])

		return total

	def _load_embeddings(self):

		try:
			self.logger.info("Loading pyltp flag embedding model")
			self.pyltp_flag_wv = KeyedVectors.load(self.pyltp_flag_embed_path)
		except Exception:

			self.logger.info("flag embedding model not found !")

		try:
			self.logger.info("Loading pyltp token embedding model")
			self.pyltp_token_wv = KeyedVectors.load(self.pyltp_token_embed_path)
		except Exception:
			self.logger.info("pyltp token embedding model not found !")

		try:
			self.logger.info("Loading jieba flag embedding model")
			self.jieba_flag_wv = KeyedVectors.load(self.jieba_flag_embed_path)
		except Exception:

			self.logger.info("jieba flag embedding model not found !")

		try:
			self.logger.info("Loading token embedding model")
			self.jieba_token_wv = KeyedVectors.load(self.jieba_token_embed_path)
		except Exception:
			self.logger.info("jieba token embedding model not found !")

		self.logger.info("Loading elmo embedding model")
		self.elmo_dict, self.elmo_embed = get_elmo_vocab(self.elmo_vocab_path, self.elmo_embed_path)

	def _get_vocabs(self):

		self.pyltp_token_wv.index2word.insert(0, '<unk>')
		self.pyltp_token_wv.index2word.insert(0, '<pad>')
		self.pyltp_token_wv.vectors = np.concatenate(
			[[np.zeros(self.pyltp_token_wv.vector_size, dtype=np.float32),
			  np.ones(self.pyltp_token_wv.vector_size, dtype=np.float32) * 0.05, ], self.pyltp_token_wv.vectors], axis=0)

		self.pyltp_flag_wv.index2word.insert(0, '<unk>')
		self.pyltp_flag_wv.index2word.insert(0, '<pad>')
		self.pyltp_flag_wv.vectors = np.concatenate(
			[[np.zeros(self.pyltp_flag_wv.vector_size, dtype=np.float32),
			  np.ones(self.pyltp_flag_wv.vector_size, dtype=np.float32) * 0.05, ], self.pyltp_flag_wv.vectors], axis=0)

		self.pyltp_flag_vocab = Vocab(self.pyltp_flag_wv.index2word, self.pyltp_flag_wv.vectors)

		self.logger.info('the final pyltp flag vocab size is {}'.format(self.pyltp_flag_vocab.size()))

		self.pyltp_token_vocab = Vocab(self.pyltp_token_wv.index2word, self.pyltp_token_wv.vectors)

		self.logger.info('the final pyltp token vocab size is {}'.format(self.pyltp_token_vocab.size()))

		self.jieba_token_wv.index2word.insert(0, '<unk>')
		self.jieba_token_wv.index2word.insert(0, '<pad>')
		self.jieba_token_wv.vectors = np.concatenate(
			[[np.zeros(self.jieba_token_wv.vector_size, dtype=np.float32),
			  np.ones(self.jieba_token_wv.vector_size, dtype=np.float32) * 0.05, ], self.jieba_token_wv.vectors],
			axis=0)

		self.jieba_flag_wv.index2word.insert(0, '<unk>')
		self.jieba_flag_wv.index2word.insert(0, '<pad>')
		self.jieba_flag_wv.vectors = np.concatenate(
			[[np.zeros(self.jieba_flag_wv.vector_size, dtype=np.float32),
			  np.ones(self.jieba_flag_wv.vector_size, dtype=np.float32) * 0.05, ], self.jieba_flag_wv.vectors], axis=0)

		self.jieba_flag_vocab = Vocab(self.jieba_flag_wv.index2word, self.jieba_flag_wv.vectors)

		self.logger.info('the final jieba flag vocab size is {}'.format(self.jieba_flag_vocab.size()))

		self.jieba_token_vocab = Vocab(self.jieba_token_wv.index2word, self.jieba_token_wv.vectors)

		self.logger.info('the final jieba token vocab size is {}'.format(self.jieba_token_vocab.size()))

		self.elmo_vocab = Vocab(list(self.elmo_dict.keys()), self.elmo_embed)

		del self.elmo_dict, self.elmo_embed
		del self.pyltp_flag_wv, self.pyltp_token_wv, self.jieba_flag_wv, self.jieba_token_wv
		gc.collect()

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

					  'start_id': [],
					  'end_id': [],
					  'qtype_vecs': [],
					  # 'answer_tokens_len': [],
					  'question_c_len': [],
					  'article_c_len': [],

					  # delta stuff
					  'delta_token_starts': [],
					  'delta_token_ends': [],
					  'delta_rouges': [],
					  'delta_span_idxs': [],

					  # hand features
					  'wiqB': [],

					  'article_pad_len': 0,
					  'question_pad_len': 0,
					  }
		for sidx, sample in enumerate(batch_data['raw_data']):


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

		for sidx, sample in enumerate(batch_data['raw_data']):
			if 'answer_tokens' in sample and len(sample['answer_tokens']):
				# batch_data['answer_tokens_len'].append(len(sample['answer_tokens']))
				batch_data['start_id'].append(sample['answer_token_start'])
				batch_data['end_id'].append(sample['answer_token_end'])
				# delta stuff
				batch_data['delta_token_starts'].extend(sample['delta_token_starts'])
				batch_data['delta_token_ends'].extend(sample['delta_token_ends'])
				batch_data['delta_rouges'].extend(sample['delta_rouges'])
				batch_data['delta_span_idxs'].extend([sidx] * len(sample['delta_rouges']))
			else:
				# fake span for some samples, only valid for testing
				batch_data['start_id'].append(0)
				batch_data['end_id'].append(0)
				# delta stuff
				batch_data['delta_token_starts'].extend([0])
				batch_data['delta_token_ends'].extend([0])
				batch_data['delta_rouges'].extend([0])
				batch_data['delta_span_idxs'].extend([sidx] * len(sample['delta_rouges']))
		# batch_data['answer_tokens_len'].append(0)
		batch_data = self._gen_hand_features(batch_data)
		return batch_data

	def _dynamic_padding(self, batch_data):
		"""
		Dynamically pads the batch_data with pad_id
		"""

		pad_id_t = self.token_vocab.get_id(self.token_vocab.pad_token)
		pad_id_f = self.flag_vocab.get_id(self.flag_vocab.pad_token)
		pad_id_e = self.flag_vocab.get_id(self.elmo_vocab.pad_token)

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
		token2idx = self.token_vocab.token2id
		flag2idx = self.flag_vocab.token2id
		elmo2idx = self.elmo_vocab.token2id

		for data_set in [self.train_set, self.test_set]:
			if data_set is None:
				continue
			for sample in data_set:


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

	def switch(self):
		ft = ['jieba', 'pyltp'] if self.use_jieba else ['jieba', 'pyltp']
		self.logger.info("Switching dataset from {} to {}".format(ft[0], ft[1]))
		self.__init__(self.pyltp_cfg, self.jieba_cfg, not self.use_jieba, self.is_test)


if __name__ == '__main__':
	from config import jieba_data_config
	from config import pyltp_data_config

	jieba_cfg = jieba_data_config.config
	pyltp_cfg = pyltp_data_config.config

	data = MilitaryAiDataset(pyltp_cfg, jieba_cfg)
