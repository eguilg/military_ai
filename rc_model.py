# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.bleu import Bleu
from utils.rouge import RougeL
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from conv_layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, char_vocab, token_vocab, flag_vocab, elmo_vocab, args, qtype_count=10):

        # logging
        self.logger = logging.getLogger("Military AI")

        # basic config
        self.algo = args.algo
        # self.suffix = args.suffix
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.use_char_emb = args.use_char_emb
        self.qtype_count = qtype_count
        self.is_train = True
        # length limit
        # self.max_p_num = args.max_p_num
        # self.max_p_len = args.max_p_len
        # self.max_q_len = args.max_q_len
        # self.max_a_len = args.max_a_len

        # the vocab
        self.char_vocab = char_vocab
        self.token_vocab = token_vocab
        self.flag_vocab = flag_vocab
        self.elmo_vocab = elmo_vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

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
        self.p_c = tf.placeholder(tf.int32, [None, None, None])
        self.q_c = tf.placeholder(tf.int32, [None, None, None])
        self.p_t_length = tf.placeholder(tf.int32, [None])
        self.q_t_length = tf.placeholder(tf.int32, [None])
        self.p_c_length = tf.placeholder(tf.int32, [None])
        self.q_c_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.wiqB = tf.placeholder(tf.float32, [None, None, 1])
        self.qtype_vec = tf.placeholder(tf.float32, [None, self.qtype_count])
        # self.wiqW = tf.placeholder(tf.float32, [None, None, 1])
        self.p_pad_len = tf.placeholder(tf.int32)
        self.q_pad_len = tf.placeholder(tf.int32)
        self.p_CL = tf.placeholder(tf.int32)
        self.q_CL = tf.placeholder(tf.int32)

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.variable_scope('token_embedding'):
            with tf.device('/cpu:0'):
                self.token_embeddings = tf.get_variable(
                    'token_embedding',
                    shape=(self.token_vocab.size(), self.token_vocab.embed_dim),
                    initializer=tf.constant_initializer(self.token_vocab.embeddings),
                    trainable=False
                )
            self.p_t_emb = tf.nn.embedding_lookup(self.token_embeddings, self.p_t)
            self.q_t_emb = tf.nn.embedding_lookup(self.token_embeddings, self.q_t)

        with tf.variable_scope('flag_embedding'):
            with tf.device('/cpu:0'):
                self.flag_embeddings = tf.get_variable(
                    'flag_embedding',
                    shape=(self.flag_vocab.size(), self.flag_vocab.embed_dim),
                    initializer=tf.constant_initializer(self.flag_vocab.embeddings),
                    trainable=False
                )
            p_f_emb = tf.nn.embedding_lookup(self.flag_embeddings, self.p_f)
            q_f_emb = tf.nn.embedding_lookup(self.flag_embeddings, self.q_f)

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

        if self.use_char_emb:
            with tf.variable_scope('char_embedding'):
                with tf.device('/cpu:0'):
                    self.char_embeddings = tf.get_variable(
                        'char_embedding',
                        shape=(self.char_vocab.size(), self.char_vocab.embed_dim),
                        initializer=tf.constant_initializer(self.char_vocab.embeddings),
                        trainable=False
                    )

                self.p_c_emb = tf.nn.embedding_lookup(self.char_embeddings, self.p_c)
                self.q_c_emb = tf.nn.embedding_lookup(self.char_embeddings, self.q_c)
                batch_size = tf.shape(self.start_label)[0]
                self.p_c_emb = tf.reshape(self.p_c_emb, [batch_size * self.p_pad_len, self.p_CL, self.char_vocab.embed_dim])
                self.q_c_emb = tf.reshape(self.q_c_emb, [batch_size * self.q_pad_len, self.q_CL, self.char_vocab.embed_dim])
                # if self.use_dropout:
                #     self.p_c_emb = tf.nn.dropout(self.p_c_emb, self.dropout_keep_prob)
                #     self.q_c_emb = tf.nn.dropout(self.q_c_emb, self.dropout_keep_prob)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('encode'):
            batch_size = tf.shape(self.start_label)[0]
            with tf.variable_scope('passage_encoding'):
                with tf.variable_scope('token_level'):
                    sep_p_t_encodes, _ = rnn('bi-lstm', self.p_t_emb, self.p_t_length, self.hidden_size)
                if self.use_char_emb:
                    with tf.variable_scope('char_level'):
                        _, sep_p_c_encodes = rnn('bi-lstm', self.p_c_emb, self.p_c_length, self.hidden_size)

                        sep_p_c_encodes = tf.reshape(sep_p_c_encodes, [batch_size, self.p_pad_len,
                                                                       self.hidden_size * 2])
                    self.sep_p_encodes = tf.concat([sep_p_t_encodes, sep_p_c_encodes], axis=-1)
                else:
                    self.sep_p_encodes = sep_p_t_encodes
            with tf.variable_scope('question_encoding'):
                with tf.variable_scope('token_level'):
                    sep_q_t_encodes, _ = rnn('bi-lstm', self.q_t_emb, self.q_t_length, self.hidden_size)
                if self.use_char_emb:
                    with tf.variable_scope('char_level'):
                        _, sep_q_c_encodes = rnn('bi-lstm', self.q_c_emb, self.q_c_length, self.hidden_size)
                        sep_q_c_encodes = tf.reshape(sep_q_c_encodes, [batch_size, self.q_pad_len,
                                                                       self.hidden_size * 2])
                    self.sep_q_encodes = tf.concat([sep_q_t_encodes, sep_q_c_encodes], axis=-1)
                else:
                    self.sep_q_encodes = sep_q_t_encodes
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

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_t_length,
                                         self.hidden_size, layer_num=1)
            # if self.algo == 'MLSTM':
            #     match_layer = MatchLSTMLayer(self.hidden_size)
            # elif self.algo == 'BIDAF':
            #     match_layer = AttentionFlowMatchLayer(self.hidden_size)
            # else:
            #     raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
            #
            # self.fuse_p_encodes, _ = match_layer.match(self.match_p_encodes, self.match_p_encodes,
            #                                            self.p_t_length, self.p_t_length)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)




    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        # with tf.variable_scope('same_question_concat'):
        #     batch_size = tf.shape(self.start_label)[0]
        #     concat_passage_encodes = tf.reshape(
        #         self.fuse_p_encodes,
        #         [batch_size, -1, 2 * self.hidden_size]
        #     )
        #     question_encodes = tf.reshape(
        #         self.sep_q_encodes,
        #         [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
        #     )
        with tf.variable_scope('decode'):
            decoder = PointerNetDecoder(self.hidden_size)
            self.start_probs, self.end_probs = decoder.decode(self.fuse_p_encodes,
                                                              self.sep_q_encodes)

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

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.main_loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.is_train:
            with tf.variable_scope("qtype-clf"):
                last_output = self.sep_q_encodes[:, -1, :]
                softmax_w = tf.get_variable("softmax_w",
                                            [last_output.get_shape().as_list()[-1], self.qtype_count],
                                            dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [self.qtype_count], dtype=tf.float32)
                type_logits = tf.matmul(last_output, softmax_w) + softmax_b

                self.type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.qtype_vec, logits=type_logits))
            self.loss = tf.add(self.main_loss, 0.2 * self.type_loss)
        else:
            self.loss = self.main_loss
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        lr = self.learning_rate
        if self.lr_decay < 1:
            global_step = tf.contrib.framework.get_or_create_global_step()
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
            self.optimizer = tf.train.RMSPropOptimizer(lr)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss, total_main_loss = 0, 0, 0
        log_every_n_batch, n_batch_loss, n_batch_main_loss = 50, 0, 0
        for bitx, batch in enumerate(train_batches, 1):
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

                         self.dropout_keep_prob: dropout_keep_prob}
            if self.use_char_emb:
                feed_dict.update(
                        {self.p_c: batch['article_char_ids'],
                         self.q_c: batch['question_char_ids'],
                         self.p_c_length: batch['article_c_len'],
                         self.q_c_length: batch['question_c_len'],
                         self.p_CL: batch['article_CL'],
                         self.q_CL: batch['question_CL']
                         })
            # print('article CL:{}\n'
            #       'question_CL:{}\n'
            #       'article pad len:{}\n'
            #       'question pad len:{}'.format(batch['article_CL'], batch['question_CL'],
            #                                    batch['article_pad_len'], batch['question_pad_len']))
            # print(batch['question_char_ids'])
            _, loss, main_loss = self.sess.run([self.train_op, self.loss, self.main_loss], feed_dict)
            batch_size = len(batch['raw_data'])
            total_loss += loss * batch_size
            total_main_loss += main_loss * batch_size
            total_num += batch_size
            n_batch_loss += loss
            n_batch_main_loss += main_loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is Total Loss: {}, Main Loss: {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch,
                    n_batch_main_loss / log_every_n_batch))
                n_batch_loss = 0
                n_batch_main_loss = 0
        return 1.0 * total_loss / total_num, 1.0 * total_main_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
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
        self.is_train = True
        max_rouge_l = 0
        for epoch in tqdm(range(1, epochs + 1)):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, shuffle=True)
            train_loss, train_main_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is Total Loss: {}, Main Loss: {}'.format(epoch, train_loss, train_main_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                    eval_loss, eval_main_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval main loss {}'.format(eval_main_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        self.save(save_dir, save_prefix)
                        max_rouge_l = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
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
        total_loss, total_main_loss, total_num = 0, 0, 0
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

                         self.dropout_keep_prob: 1.0}

            if self.use_char_emb:
                feed_dict.update(
                    {self.p_c: batch['article_char_ids'],
                     self.q_c: batch['question_char_ids'],
                     self.p_c_length: batch['article_c_len'],
                     self.q_c_length: batch['question_c_len'],
                     self.p_CL: batch['article_CL'],
                     self.q_CL: batch['question_CL']
                     })
            start_probs, end_probs, loss, main_loss = self.sess.run([self.start_probs,
                                                                     self.end_probs, self.loss, self.main_loss],
                                                                    feed_dict)
            batch_size = len(batch['raw_data'])
            total_loss += loss * batch_size
            total_main_loss += main_loss * batch_size
            total_num += batch_size

            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):
                best_answer = self.find_best_answer(sample, start_prob, end_prob)
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
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                # for pred_answer in pred_answers:
                #     fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
                json.dump(pred_answers, fout, ensure_ascii=False)

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        ave_main_loss = 1.0 * total_main_loss / total_num

        return ave_loss, ave_main_loss, bleu_rouge

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
        # return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
