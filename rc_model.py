# coding = utf-8
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import json
import pickle
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from layers.basic_rnn import rnn
from layers.match_layer import AttentionFlowMatchLayer
from layers.match_layer import MatchLSTMLayer
from layers.pointer_net import PointerNetDecoder
from utils.bleu import Bleu
from utils.rouge import RougeL


class RCModel(object):
	"""
	Implements the main reading comprehension model.
	"""

	def __init__(self, dataset, cfg):

		# logging
		self.logger = logging.getLogger("Military AI")

		# basic config
		self.algo = cfg.algo
		self.suffix = cfg.suffix
		self.hidden_size = cfg.hidden_size

		self.optim_type = cfg.optim
		self.learning_rate = cfg.learning_rate
		self.lr_decay = cfg.lr_decay
		self.loss_type = cfg.loss_type
		self.weight_decay = cfg.weight_decay
		self.use_dropout = cfg.dropout_keep_prob < 1

		self.switch = cfg.switch

		self.qtype_count = cfg.qtype_count

		# the vocab
		self.pyltp_token_vocab = dataset.pyltp_token_vocab
		self.pyltp_flag_vocab = dataset.pyltp_flag_vocab

		self.jieba_token_vocab = dataset.jieba_token_vocab
		self.jieba_flag_vocab = dataset.jieba_flag_vocab
		self.data = dataset
		self.model_name = self.algo + self.suffix + '_cv' + str(self.data.cv)

		self.elmo_vocab = dataset.elmo_vocab

		# session info
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=sess_config)

		self._build_graph()

		# save info
		self.saver = tf.train.Saver(max_to_keep=12)

		# initialize the model

		self.sess.run(tf.global_variables_initializer())
		self.logger.info('Model Initialized')

	def _build_graph(self):
		"""
		Builds the computation graph with Tensorflow
		"""
		start_t = time.time()
		self._setup_placeholders()
		self._embed()
		self._encode()
		self._match()
		self._hand_feature()
		self._self_att()
		self._fuse()
		self._decode()
		self._compute_loss()
		self._create_train_op()
		self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
		param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
		self.logger.info('There are {} parameters in the model'.format(param_num))

	def _setup_placeholders(self):
		"""
		Placeholders
		"""
		self.p_t = tf.placeholder(tf.int32, [None, None])
		self.q_t = tf.placeholder(tf.int32, [None, None])
		self.p_f = tf.placeholder(tf.int32, [None, None])
		self.q_f = tf.placeholder(tf.int32, [None, None])
		self.p_e = tf.placeholder(tf.int32, [None, None])
		self.q_e = tf.placeholder(tf.int32, [None, None])

		self.p_t_length = tf.placeholder(tf.int32, [None])
		self.q_t_length = tf.placeholder(tf.int32, [None])

		self.start_label = tf.placeholder(tf.int32, [None])
		self.end_label = tf.placeholder(tf.int32, [None])
		# delta stuff
		self.delta_starts = tf.placeholder(tf.int32, [None])
		self.delta_ends = tf.placeholder(tf.int32, [None])
		self.delta_span_idxs = tf.placeholder(tf.int32, [None])
		self.delta_rouges = tf.placeholder(tf.float32, [None])

		self.wiqB = tf.placeholder(tf.float32, [None, None, 1])
		self.qtype_vec = tf.placeholder(tf.float32, [None, self.qtype_count])

		self.p_pad_len = tf.placeholder(tf.int32)
		self.q_pad_len = tf.placeholder(tf.int32)

		self.use_jieba = tf.placeholder(tf.bool, shape=[])

		self.dropout_keep_prob = tf.placeholder(tf.float32)
		self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

	def _embed(self):
		"""
		The embedding layer, question and passage share embeddings
		"""
		with tf.variable_scope('token_embedding'):
			with tf.variable_scope('jieba'):
				with tf.device('/cpu:0'):
					self.jieba_token_embeddings = tf.get_variable(
						'token_embedding',
						shape=(self.jieba_token_vocab.size(), self.jieba_token_vocab.embed_dim),
						initializer=tf.constant_initializer(self.jieba_token_vocab.embeddings),
						trainable=False
					)
			with tf.variable_scope('pyltp'):
				with tf.device('/cpu:0'):
					self.pyltp_token_embeddings = tf.get_variable(
						'token_embedding',
						shape=(self.pyltp_token_vocab.size(), self.pyltp_token_vocab.embed_dim),
						initializer=tf.constant_initializer(self.pyltp_token_vocab.embeddings),
						trainable=False
					)
			self.p_t_emb = tf.cond(self.use_jieba,
								   true_fn=lambda: tf.nn.embedding_lookup(self.jieba_token_embeddings, self.p_t),
								   false_fn=lambda: tf.nn.embedding_lookup(self.pyltp_token_embeddings, self.p_t))

			self.q_t_emb = tf.cond(self.use_jieba,
								   true_fn=lambda: tf.nn.embedding_lookup(self.jieba_token_embeddings, self.q_t),
								   false_fn=lambda: tf.nn.embedding_lookup(self.pyltp_token_embeddings, self.q_t))

		with tf.variable_scope('flag_embedding'):
			with tf.variable_scope('jieba'):
				with tf.device('/cpu:0'):
					self.jieba_flag_embeddings = tf.get_variable(
						'flag_embedding',
						shape=(self.jieba_flag_vocab.size(), self.jieba_flag_vocab.embed_dim),
						initializer=tf.constant_initializer(self.jieba_flag_vocab.embeddings),
						trainable=True
					)
			with tf.variable_scope('pyltp'):
				with tf.device('/cpu:0'):
					self.pyltp_flag_embeddings = tf.get_variable(
						'flag_embedding',
						shape=(self.pyltp_flag_vocab.size(), self.pyltp_flag_vocab.embed_dim),
						initializer=tf.constant_initializer(self.pyltp_flag_vocab.embeddings),
						trainable=True
					)
			p_f_emb = tf.cond(self.use_jieba,
							  true_fn=lambda: tf.nn.embedding_lookup(self.jieba_flag_embeddings, self.p_f),
							  false_fn=lambda: tf.nn.embedding_lookup(self.pyltp_flag_embeddings, self.p_f))

			q_f_emb = tf.cond(self.use_jieba,
							  true_fn=lambda: tf.nn.embedding_lookup(self.jieba_flag_embeddings, self.q_f),
							  false_fn=lambda: tf.nn.embedding_lookup(self.pyltp_flag_embeddings, self.q_f))

		with tf.variable_scope('elmo_embedding'):
			with tf.device('/cpu:0'):
				self.elmo_embeddings = tf.get_variable(
					'elmo_embedding',
					shape=(self.elmo_vocab.size(), self.elmo_vocab.embed_dim),
					initializer=tf.constant_initializer(self.elmo_vocab.embeddings),
					trainable=False
				)
			p_e_emb = tf.nn.embedding_lookup(self.elmo_embeddings, self.p_e)
			q_e_emb = tf.nn.embedding_lookup(self.elmo_embeddings, self.q_e)

		self.p_t_emb = tf.concat([self.p_t_emb, p_f_emb, p_e_emb], axis=-1)
		self.q_t_emb = tf.concat([self.q_t_emb, q_f_emb, q_e_emb], axis=-1)

	# if self.use_dropout:
	#     self.p_t_emb = tf.nn.dropout(self.p_t_emb, self.dropout_keep_prob)
	#     self.q_t_emb = tf.nn.dropout(self.q_t_emb, self.dropout_keep_prob)

	# if self.use_dropout:
	#     self.p_c_emb = tf.nn.dropout(self.p_c_emb, self.dropout_keep_prob)
	#     self.q_c_emb = tf.nn.dropout(self.q_c_emb, self.dropout_keep_prob)

	def _encode(self):
		"""
		Employs two Bi-LSTMs to encode passage and question separately
		"""
		with tf.variable_scope('encode'):
			with tf.variable_scope('passage_encoding'):
				with tf.variable_scope('token_level'):
					self.sep_p_encodes, _ = rnn('bi-lstm', self.p_t_emb, self.p_t_length, self.hidden_size)
			with tf.variable_scope('question_encoding'):
				with tf.variable_scope('token_level'):
					self.sep_q_encodes, _ = rnn('bi-lstm', self.q_t_emb, self.q_t_length, self.hidden_size)
			if self.use_dropout:
				self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
				self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

	def _match(self):
		"""
		The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
		"""
		with tf.variable_scope('match'):
			if self.algo == 'MLSTM':
				match_layer = MatchLSTMLayer(self.hidden_size)
			elif self.algo == 'BIDAF':
				match_layer = AttentionFlowMatchLayer(self.hidden_size)
			else:
				raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
			self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
														self.p_t_length, self.q_t_length)

			self.match_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_t_length,
										  self.hidden_size, layer_num=1)

			if self.use_dropout:
				self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

	def _hand_feature(self):
		"""
		Concats hand features
		"""
		with tf.variable_scope('hand_feature'):
			batch_size = tf.shape(self.start_label)[0]
			self.wiqB = tf.reshape(self.wiqB, [batch_size, self.p_pad_len, 1])
			self.match_p_encodes = tf.concat([self.match_p_encodes, self.wiqB], axis=-1)

	def _self_att(self):
		"""
		Self attention layer
		"""
		with tf.variable_scope('self_att'):
			# if self.algo == 'MLSTM':
			# 	self_att_layer = MatchLSTMLayer(self.hidden_size)
			# elif self.algo == 'BIDAF':
			self_att_layer = AttentionFlowMatchLayer(self.hidden_size)
			# else:
			# 	raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
			self.self_att_p_encodes, _ = self_att_layer.match(self.match_p_encodes, self.match_p_encodes,
															  self.p_t_length, self.p_t_length)
			if self.use_dropout:
				self.self_att_p_encodes = tf.nn.dropout(self.self_att_p_encodes, self.dropout_keep_prob)

	def _fuse(self):
		"""
		Employs Bi-LSTM again to fuse the context information after match layer
		"""
		with tf.variable_scope('fusion'):
			self.fuse_p_encodes, _ = rnn('bi-lstm', self.self_att_p_encodes, self.p_t_length,
										 self.hidden_size, layer_num=1)

			if self.use_dropout:
				self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

	def _decode(self):
		"""
		Employs Pointer Network to get the the probs of each position
		to be the start or end of the predicted answer.
		Note that we concat the fuse_p_encodes for the passages in the same document.
		And since the encodes of queries in the same document is same, we select the first one.
		"""

		with tf.variable_scope('decode'):
			decoder = PointerNetDecoder(self.hidden_size)
			self.start_probs, self.end_probs = decoder.decode(self.fuse_p_encodes, self.sep_q_encodes)
			self.out_matrix = tf.matmul(tf.expand_dims(self.start_probs, axis=2),
										tf.expand_dims(self.end_probs, axis=1))
			outer = tf.matrix_band_part(self.out_matrix, 0, -1)
			self.pred_starts = tf.argmax(tf.reduce_max(outer, axis=2), axis=-1)
			self.pred_ends = tf.argmax(tf.reduce_max(outer, axis=1), axis=-1)

	def _compute_loss(self):
		"""
		The loss function
		"""

		def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
			"""
			negative log likelyhood loss
			"""
			with tf.name_scope(scope, "log_loss"):
				labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
				losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
			return losses

		with tf.variable_scope("pointer_loss"):
			self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
			self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
			self.all_params = tf.trainable_variables()
			self.pointer_loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))

		with tf.variable_scope("type_loss"):
			last_output = self.sep_q_encodes[:, -1, :]
			softmax_w = tf.get_variable("softmax_w",
										[last_output.get_shape().as_list()[-1], self.qtype_count],
										dtype=tf.float32)
			softmax_b = tf.get_variable("softmax_b", [self.qtype_count], dtype=tf.float32)
			type_logits = tf.matmul(last_output, softmax_w) + softmax_b

			self.type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
				labels=self.qtype_vec, logits=type_logits))

		with tf.variable_scope("mrl"):
			batch_size = tf.shape(self.start_label)[0]
			out_matrix = tf.reshape(self.out_matrix, [batch_size, self.p_pad_len ** 2])
			# hard
			delta_hard_pos = self.start_label * self.p_pad_len + self.end_label
			batch_idx_hard = tf.range(0, batch_size)
			indices_hard = tf.to_int64(tf.stack([batch_idx_hard, delta_hard_pos], axis=1))
			delta_hard = 1.0 - tf.sparse_to_dense(indices_hard,
												  tf.to_int64([batch_size, self.p_pad_len ** 2]),
												  1.0, 0.0)

			# soft
			delta_soft_pos = self.delta_starts * self.p_pad_len + self.delta_ends
			indices_soft = tf.to_int64(tf.stack([self.delta_span_idxs, delta_soft_pos], axis=1))
			delta_soft = 1.0 - tf.sparse_to_dense(indices_soft,
												  tf.to_int64([batch_size, self.p_pad_len ** 2]),
												  self.delta_rouges, 0.0)

			delta_mix = tf.reduce_min(tf.stack([delta_hard, delta_soft], axis=2), axis=-1)

			self.mrl_hard = tf.reduce_mean(tf.reduce_sum(delta_hard * out_matrix, axis=-1))
			self.mrl_soft = tf.reduce_mean(tf.reduce_sum(delta_soft * out_matrix, axis=-1))
			self.mrl_mix = tf.reduce_mean(tf.reduce_sum(delta_mix * out_matrix, axis=-1))

		if self.loss_type == 'pointer':
			self.mrl = tf.constant(0, dtype=tf.float32)
			self.loss = self.pointer_loss + 0.15 * self.type_loss
		elif self.loss_type == 'mrl_mix':
			self.mrl = self.mrl_mix
			self.loss = self.mrl + 0.15 * self.type_loss
		elif self.loss_type == 'mrl_soft':
			self.mrl = self.mrl_soft
			self.loss = self.mrl + 0.15 * self.type_loss
		elif self.loss_type == 'mrl_hard':
			self.mrl = self.mrl_hard
			self.loss = self.mrl + 0.15 * self.type_loss
		else:
			assert 0 != 0

		if self.weight_decay > 0:
			with tf.variable_scope('l2_loss'):
				l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
			self.loss += self.weight_decay * l2_loss

	def _create_train_op(self):
		"""
		Selects the training algorithm and creates a train operation with it
		"""
		lr = self.learning_rate
		global_step = tf.train.get_or_create_global_step()
		if self.lr_decay < 1:
			self.decay_learning_rate = tf.train.exponential_decay(
				self.learning_rate,
				global_step,
				300,
				self.lr_decay
			)
			lr = self.decay_learning_rate

		if self.optim_type == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(lr)
		elif self.optim_type == 'adam':
			self.optimizer = tf.train.AdamOptimizer(lr)
		elif self.optim_type == 'rprop':
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.lr_decay)
		elif self.optim_type == 'adadelta':
			self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, self.lr_decay)
		elif self.optim_type == 'sgd':
			self.optimizer = tf.train.GradientDescentOptimizer(lr)
		else:
			raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
		self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

	def _train_epoch(self, train_batches, dropout_keep_prob,
					 max_rouge_l, save_dir):
		"""
		Trains the model for a single epoch.
		Args:
			train_batches: iterable batch data for training
			dropout_keep_prob: float value indicating dropout keep probability
		"""
		cur_dataset_fn = lambda use_jieba: 'jieba' if use_jieba else 'pyltp'
		total_num, total_mrl, total_pointer_loss = 0, 0, 0
		flag = 1
		log_every_n_batch, eval_every_n_batch, n_batch_mrl, n_batch_pointer_loss = 50, 1000, 0, 0
		for bitx, batch in enumerate(train_batches, 1):

			if max(max_rouge_l.values()) >= 0.885:
				flag = 2
			if max(max_rouge_l.values()) >= 0.89:
				flag = 3
			if max(max_rouge_l.values()) >= 0.895:
				flag = 4
			feed_dict = {self.p_t: batch['article_token_ids'],
						 self.q_t: batch['question_token_ids'],
						 self.p_f: batch['article_flag_ids'],
						 self.q_f: batch['question_flag_ids'],
						 self.p_e: batch['article_elmo_ids'],
						 self.q_e: batch['question_elmo_ids'],
						 self.p_pad_len: batch['article_pad_len'],
						 self.q_pad_len: batch['question_pad_len'],
						 self.p_t_length: batch['article_tokens_len'],
						 self.q_t_length: batch['question_tokens_len'],

						 self.start_label: batch['start_id'],
						 self.end_label: batch['end_id'],
						 self.wiqB: batch['wiqB'],
						 self.qtype_vec: batch['qtype_vecs'],
						 # delta stuff
						 self.delta_starts: batch['delta_token_starts'],
						 self.delta_ends: batch['delta_token_ends'],
						 self.delta_span_idxs: batch['delta_span_idxs'],
						 self.delta_rouges: batch['delta_rouges'],

						 self.use_jieba: self.data.use_jieba,

						 self.dropout_keep_prob: dropout_keep_prob}

			_, mrl, pointer_loss, type_loss = self.sess.run(
				[self.train_op, self.mrl, self.pointer_loss, self.type_loss], feed_dict)
			batch_size = len(batch['raw_data'])
			total_mrl += mrl * batch_size
			total_pointer_loss += pointer_loss * batch_size
			total_num += batch_size
			n_batch_mrl += mrl
			n_batch_pointer_loss += pointer_loss

			if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
				self.logger.info('Average loss from batch {} to {} is MRL Loss: {}, Pointer Loss: {}'.format(
					bitx - log_every_n_batch + 1, bitx, n_batch_mrl / log_every_n_batch,
					n_batch_pointer_loss / log_every_n_batch))
				n_batch_mrl = 0
				n_batch_pointer_loss = 0
			if (eval_every_n_batch // flag) > 0 and bitx > 0 and bitx % (eval_every_n_batch // flag) == 0:
				self.logger.info('Evaluating the model after batch {}'.format(bitx))
				if self.data.dev_set is not None:
					eval_batches = self.data.gen_mini_batches('dev', batch_size, shuffle=False)
					eval_mrl, eval_pointer_loss, bleu_rouge = self.evaluate(eval_batches)
					self.logger.info('Dev eval mrl {}'.format(eval_mrl))
					self.logger.info('Dev eval pointer loss {}'.format(eval_pointer_loss))
					self.logger.info('Dev eval result: {}'.format(bleu_rouge))

					rouge_key = cur_dataset_fn(self.data.use_jieba)
					if bleu_rouge['Rouge-L'] > max_rouge_l[rouge_key]:
						if bleu_rouge['Rouge-L'] < 0.885:
							self.save(save_dir,
									  self.loss_type + '_' + rouge_key + '_' + str(round(bleu_rouge['Rouge-L'], 4)))
						else:
							self.save(save_dir, self.loss_type + '_' + rouge_key + '_best')
						max_rouge_l[rouge_key] = bleu_rouge['Rouge-L']
				else:
					self.logger.warning('No dev set is loaded for evaluation in the dataset!')

		return 1.0 * total_mrl / total_num, 1.0 * total_pointer_loss / total_num, max_rouge_l

	def train(self, epochs, batch_size, save_dir,
			  dropout_keep_prob=1.0, evaluate=True):
		"""
		Train the model with data
		Args:
			data: the BRCDataset class implemented in dataset.py
			epochs: number of training epochs
			batch_size:
			save_dir: the directory to save the model
			save_prefix: the prefix indicating the model type
			dropout_keep_prob: float value indicating dropout keep probability
			evaluate: whether to evaluate the model on test set after each epoch
		"""
		max_rouge_l = {
			'jieba': 0.0,
			'pyltp': 0.0
		}
		cur_dataset_fn = lambda use_jieba: 'jieba' if use_jieba else 'pyltp'
		for epoch in tqdm(range(1, epochs + 1)):
			self.logger.info('Training the model for epoch {}'.format(epoch))
			train_batches = self.data.gen_mini_batches('train', batch_size, shuffle=True)
			train_mrl, train_pointer_loss, max_rouge_l = self._train_epoch(train_batches, dropout_keep_prob,
																		   max_rouge_l, save_dir)
			self.logger.info(
				'Average train loss for epoch {} is MRL Loss: {}, Pointer Loss: {}'.format(epoch, train_mrl,
																						   train_pointer_loss))
			rouge_key = cur_dataset_fn(self.data.use_jieba)
			if evaluate:
				self.logger.info('Evaluating the model after epoch {}'.format(epoch))
				if self.data.dev_set is not None:
					eval_batches = self.data.gen_mini_batches('dev', batch_size, shuffle=False)
					eval_mrl, eval_pointer_loss, bleu_rouge = self.evaluate(eval_batches)
					self.logger.info('Dev eval mrl {}'.format(eval_mrl))
					self.logger.info('Dev eval pointer loss {}'.format(eval_pointer_loss))
					self.logger.info('Dev eval result: {}'.format(bleu_rouge))

					if bleu_rouge['Rouge-L'] > max_rouge_l[rouge_key]:
						if bleu_rouge['Rouge-L'] < 0.885:
							self.save(save_dir,
									  self.loss_type + '_' + rouge_key + '_' + str(round(bleu_rouge['Rouge-L'], 4)))
						else:
							self.save(save_dir, self.loss_type + '_' + rouge_key + '_best')
						max_rouge_l[rouge_key] = bleu_rouge['Rouge-L']
				else:
					self.logger.warning('No dev set is loaded for evaluation in the dataset!')
			else:
				self.save(save_dir, rouge_key + '_' + str(epoch))

			if self.switch:
				self.data.switch()

	def evaluate(self, eval_batches, result_dir=None, result_prefix=None):
		"""
		Evaluates the model performance on eval_batches and results are saved if specified
		Args:
			eval_batches: iterable batch data
			result_dir: directory to save predicted answers, answers will not be saved if None
			result_prefix: prefix of the file for saving predicted answers,
						   answers will not be saved if None
			save_full_info: if True, the pred_answers will be added to raw sample and saved
		"""
		pred_answers, ref_answers = [], []
		total_mrl, total_pointer_loss, total_num = 0, 0, 0
		rl, bleu = RougeL(), Bleu()
		ariticle_map = {}
		for b_itx, batch in enumerate(eval_batches):
			feed_dict = {self.p_t: batch['article_token_ids'],
						 self.q_t: batch['question_token_ids'],
						 self.p_f: batch['article_flag_ids'],
						 self.q_f: batch['question_flag_ids'],
						 self.p_e: batch['article_elmo_ids'],
						 self.q_e: batch['question_elmo_ids'],
						 self.p_pad_len: batch['article_pad_len'],
						 self.q_pad_len: batch['question_pad_len'],
						 self.p_t_length: batch['article_tokens_len'],
						 self.q_t_length: batch['question_tokens_len'],

						 self.start_label: batch['start_id'],
						 self.end_label: batch['end_id'],
						 self.wiqB: batch['wiqB'],
						 self.qtype_vec: batch['qtype_vecs'],

						 # delta stuff
						 self.delta_starts: batch['delta_token_starts'],
						 self.delta_ends: batch['delta_token_ends'],
						 self.delta_span_idxs: batch['delta_span_idxs'],
						 self.delta_rouges: batch['delta_rouges'],

						 self.use_jieba: self.data.use_jieba,

						 self.dropout_keep_prob: 1.0}

			pred_starts, pred_ends, prob1, prob2, mrl, pointer_loss = self.sess.run([self.pred_starts,
																					 self.pred_ends,
																					 self.start_probs,
																					 self.end_probs,
																					 self.mrl,
																					 self.pointer_loss],
																					feed_dict)
			batch_size = len(batch['raw_data'])
			total_mrl += mrl * batch_size
			total_pointer_loss += pointer_loss * batch_size
			total_num += batch_size

			for sample, best_start, best_end in zip(batch['raw_data'], pred_starts, pred_ends):
				best_answer = ''.join(sample['article_tokens'][best_start: best_end + 1])
				if sample['article_id'] not in ariticle_map:
					ariticle_map[sample['article_id']] = len(ariticle_map)
					pred_answers.append({'article_id': sample['article_id'],
										 'questions': []
										 })
					ref_answers.append({'article_id': sample['article_id'],
										'questions': []
										})

				pred_answers[ariticle_map[sample['article_id']]]['questions'].append(
					{'question_id': sample['question_id'],
					 'answer': best_answer
					 })
				ref_answers[ariticle_map[sample['article_id']]]['questions'].append(
					{'question_id': sample['question_id'],
					 'answer': sample['answer']
					 })

				rl.add_inst(best_answer, sample['answer'])
				bleu.add_inst(best_answer, sample['answer'])

		# compute the bleu and rouge scores
		rougel = rl.get_score()
		bleu4 = bleu.get_score()
		bleu_rouge = {'Rouge-L': rougel,
					  'Bleu-4': bleu4
					  }

		if result_dir is not None and result_prefix is not None:
			result_path = os.path.join(result_dir, self.model_name)
			result_file = os.path.join(result_dir, self.model_name, result_prefix + '_dev.json')
			if not os.path.isdir(result_path):
				os.makedirs(result_path)
			with open(result_file, 'w') as fout:
				# for pred_answer in pred_answers:
				#     fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
				json.dump(pred_answers, fout, ensure_ascii=False)

			self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

		# this average loss is invalid on test set, since we don't have true start_id and end_id
		ave_mrl = 1.0 * total_mrl / total_num
		ave_pointer_loss = 1.0 * total_pointer_loss / total_num

		return ave_mrl, ave_pointer_loss, bleu_rouge

	def find_best_answer(self, sample, start_probs, end_probs, passage_len=None):
		"""
		Finds the best answer with the maximum start_prob * end_prob from a single passage
		"""
		if passage_len is None:
			passage_len = len(start_probs)
		else:
			passage_len = min(len(start_probs), passage_len)
		best_start, best_end, max_prob = -1, -1, 0
		for start_idx in range(passage_len):
			for ans_len in range(1, passage_len - start_idx):
				end_idx = start_idx + ans_len - 1
				if end_idx >= passage_len:
					continue
				prob = start_probs[start_idx] * end_probs[end_idx]
				if prob > max_prob:
					best_start = start_idx
					best_end = end_idx
					max_prob = prob

		best_answer = ''.join(
			sample['article_tokens'][best_start: best_end + 1])
		return best_answer

	def predict_for_ensemble(self, eval_batches, result_dir=None, result_prefix=None):
		"""
		Evaluates the model performance on eval_batches and results are saved if specified
		Args:
			eval_batches: iterable batch data
			result_dir: directory to save predicted answers, answers will not be saved if None
			result_prefix: prefix of the file for saving predicted answers,
						   answers will not be saved if None
			save_full_info: if True, the pred_answers will be added to raw sample and saved
		"""
		answer_dict = {}

		for b_itx, batch in enumerate(eval_batches):
			feed_dict = {self.p_t: batch['article_token_ids'],
						 self.q_t: batch['question_token_ids'],
						 self.p_f: batch['article_flag_ids'],
						 self.q_f: batch['question_flag_ids'],
						 self.p_e: batch['article_elmo_ids'],
						 self.q_e: batch['question_elmo_ids'],
						 self.p_pad_len: batch['article_pad_len'],
						 self.q_pad_len: batch['question_pad_len'],
						 self.p_t_length: batch['article_tokens_len'],
						 self.q_t_length: batch['question_tokens_len'],

						 self.start_label: batch['start_id'],
						 self.end_label: batch['end_id'],
						 self.wiqB: batch['wiqB'],
						 self.qtype_vec: batch['qtype_vecs'],

						 # delta stuff
						 self.delta_starts: batch['delta_token_starts'],
						 self.delta_ends: batch['delta_token_ends'],
						 self.delta_span_idxs: batch['delta_span_idxs'],
						 self.delta_rouges: batch['delta_rouges'],

						 self.use_jieba: self.data.use_jieba,

						 self.dropout_keep_prob: 1.0}

			batch_prob1, batch_prob2 = self.sess.run([self.start_probs,
													  self.end_probs],
													 feed_dict)

			for sample, prob1, prob2 in zip(batch['raw_data'], batch_prob1, batch_prob2):
				answer_dict[sample['question_id']] = {'prob1': prob1,
													  'prob2': prob2,
													  'article_id': sample['article_id']}

		if result_dir is not None and result_prefix is not None:
			result_path = os.path.join(result_dir, self.model_name)
			result_file = os.path.join(result_dir, self.model_name, result_prefix + '_test.pkl')
			if not os.path.isdir(result_path):
				os.makedirs(result_path)
			with open(result_file, 'wb') as fout:
				# for pred_answer in pred_answers:
				#     fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
				pickle.dump(answer_dict, fout, False)

			self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))
		return

	# return (best_start, best_end), max_prob

	def save(self, model_dir, model_prefix):
		"""
		Saves the model into model_dir with model_prefix as the model indicator
		"""
		model_path = os.path.join(model_dir, self.model_name)
		if not os.path.isdir(model_path):
			os.makedirs(model_path)
		self.saver.save(self.sess, os.path.join(model_path, model_prefix))
		self.logger.info('Model saved in {}, with prefix {}.'.format(model_path, model_prefix))

	def restore(self, model_dir, model_prefix):
		"""
		Restores the model into model_dir from model_prefix as the model indicator
		"""
		model_path = os.path.join(model_dir, self.model_name)

		self.saver.restore(self.sess, os.path.join(model_path, model_prefix))
		self.logger.info('Model restored from {}, with prefix {}'.format(model_path, model_prefix))
