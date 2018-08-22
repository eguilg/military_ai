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
This module prepares and runs the whole system.
"""
import sys

# if sys.version[0] == '2':
#     # reload(sys)
#     sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import MilitaryAiDataset
from vocab import Vocab
from rc_model import RCModel

root = './'
train_raw = root + 'data/train/question.json'
train_processed = root + 'data/train/question_preprocessed.json'
# test_raw = ''
# test_processed = ''
char_embed = root + 'data/embedding/char_embed75.wv'
token_embed = root + 'data/embedding/token_embed300.wv'

elmo_dict = root + 'data/embedding/elmo-military_vocab.txt'
elmo_embed = root + 'data/embedding/elmo-military_emb.pkl'


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on MilitaryAi DataSet')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--use_embe', type=int, default=1, help='is use embeddings vector file')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--lr_decay', type=float, default=1.0,
                                help='lr decay every 300 steps')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.8,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=15,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--suffix', type=str, default='',
                                help='model file name suffix')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--use_char_emb', type=int, default=0,
                                help='if using char embeddings')
    # model_settings.add_argument('--max_p_num', type=int, default=5,
    #                             help='max passage num in one sample')
    # model_settings.add_argument('--max_p_len', type=int, default=500,
    #                             help='max length of passage')
    # model_settings.add_argument('--max_q_len', type=int, default=60,
    #                             help='max length of question')
    # model_settings.add_argument('--max_a_len', type=int, default=200,
    #                             help='max length of answer')
    model_settings.add_argument('--is_restore', type=int, default=0, help='is restore model from file')


    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_raw_files', nargs='+',
                               default=[train_raw],
                               help='list of files that contain the raw train data')
    path_settings.add_argument('--train_files', nargs='+',
                               default=[train_processed],
                               help='list of files that contain the preprocessed train data')
    # path_settings.add_argument('--dev_files', nargs='+',
    #                            default=[s_dev],
    #                            help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_raw_files', nargs='+',
                               default=[],
                               help='list of files that contain the raw test data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=[],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--char_embed_file', nargs='+',
                               default=char_embed,
                               help='char embedding file')
    path_settings.add_argument('--token_embed_file', nargs='+',
                               default=token_embed,
                               help='token embedding file')
    path_settings.add_argument('--elmo_dict_file', nargs='+',
                               default=elmo_dict,
                               help='elmo_dict_file')
    path_settings.add_argument('--elmo_embed_file', nargs='+',
                               default=elmo_embed,
                               help='elmo embedding file')
    path_settings.add_argument('--data_dir', default='./data/train',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='./data/embedding/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("Military AI")
    logger.info('Checking the data files...')
    for data_path in args.train_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    logger.info('Load data_set and vocab...')

    mai_data = MilitaryAiDataset(args.train_files, args.train_raw_files,
                                 args.test_files, args.test_raw_files,
                                 args.char_embed_file, args.token_embed_file,
                                 args.elmo_dict_file, args.elmo_embed_file,
                                 char_min_cnt=1, token_min_cnt=3)

    logger.info('Assigning embeddings...')
    if not args.use_embe:
        mai_data.token_vocab.randomly_init_embeddings(args.embed_size)
        mai_data.char_vocab.randomly_init_embeddings(args.embed_size)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("Military AI")
    logger.info('Load data_set and vocab...')

    mai_data = MilitaryAiDataset(args.train_files, args.train_raw_files,
                                 args.test_files, args.test_raw_files,
                                 args.char_embed_file, args.token_embed_file,
                                 args.elmo_dict_file, args.elmo_embed_file,
                                 char_min_cnt=1, token_min_cnt=3)

    logger.info('Assigning embeddings...')
    if not args.use_embe:
        mai_data.token_vocab.randomly_init_embeddings(args.embed_size)
        mai_data.char_vocab.randomly_init_embeddings(args.embed_size)
    logger.info('Initialize the model...')
    rc_model = RCModel(mai_data.char_vocab, mai_data.token_vocab,
                       mai_data.flag_vocab, mai_data.elmo_vocab, args)
    if args.is_restore:
        rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo+args.suffix)
    logger.info('Training the model...')
    rc_model.train(mai_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo+args.suffix,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("Military AI")
    logger.info('Load data_set and vocab...')
    mai_data = MilitaryAiDataset(args.train_files, args.train_raw_files,
                                 args.test_files, args.test_raw_files,
                                 args.char_embed_file, args.token_embed_file,
                                 args.elmo_dict_file, args.elmo_embed_file,
                                 char_min_cnt=1, token_min_cnt=3)

    logger.info('Assigning embeddings...')
    if not args.use_embe:
        mai_data.token_vocab.randomly_init_embeddings(args.embed_size)
        mai_data.char_vocab.randomly_init_embeddings(args.embed_size)
    logger.info('Restoring the model...')
    rc_model = RCModel(mai_data.char_vocab, mai_data.token_vocab,
                       mai_data.flag_vocab, mai_data.elmo_vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo+args.suffix)
    logger.info('Evaluating the model on dev set...')
    dev_batches = mai_data.gen_mini_batches('dev', args.batch_size, shuffle=False)
    dev_loss, dev_main_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_main_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("Military AI")
    logger.info('Load data_set and vocab...')
    mai_data = MilitaryAiDataset(args.train_files, args.train_raw_files,
                                 args.test_files, args.test_raw_files,
                                 args.char_embed_file, args.token_embed_file,
                                 args.elmo_dict_file, args.elmo_embed_file,
                                 char_min_cnt=1, token_min_cnt=3)
    logger.info('Assigning embeddings...')
    if not args.use_embe:
        mai_data.token_vocab.randomly_init_embeddings(args.embed_size)
        mai_data.char_vocab.randomly_init_embeddings(args.embed_size)
    logger.info('Restoring the model...')
    rc_model = RCModel(mai_data.char_vocab, mai_data.token_vocab,
                       mai_data.flag_vocab, mai_data.elmo_vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo+args.suffix)
    logger.info('Predicting answers for test set...')
    test_batches = mai_data.gen_mini_batches('test', args.batch_size, shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("Military AI")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
