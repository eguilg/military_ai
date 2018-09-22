# coding = utf-8
"""
This module prepares and runs the whole system.
"""
import os
import logging
import json
import multiprocessing
from gensim.models import Word2Vec, KeyedVectors
from preprocess import PreProcessor
from config import base_config
from config import dataset_base_config, jieba_data_config, pyltp_data_config

base_cfg = base_config.config
data_base_cfg = dataset_base_config.config
jieba_cfg = jieba_data_config.config
pyltp_cfg = pyltp_data_config.config


def preprocess_train(cfg):
	logger = logging.getLogger("Military AI")
	logger.info('Preparing ' + cfg.cut_word_method + ' cut data set...')
	logger.info('Checking raw data files...')
	assert os.path.exists(cfg.train_raw_file), '{} file does not exist.'.format(cfg.train_raw_file)
	logger.info('Preparing the directories...')
	for dir_path in [cfg.vocab_dir, cfg.summary_dir]:
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	logger.info('Checking processed train data file...')
	if not os.path.isfile(cfg.train_preprocessed_file):
		logger.info('Preprocessing train data file...')
		p = PreProcessor(cfg)
		raw_data_path = cfg.train_raw_file
		croups, flag_croups, dataset = p.preprocess_dataset(raw_data_path)
		with open(cfg.train_croups_file, 'w') as fo:
			json.dump(croups, fo)
		with open(cfg.train_flag_croups_file, 'w') as fo:
			json.dump(flag_croups, fo)
		with open(cfg.train_preprocessed_file, 'w') as fo:
			json.dump(dataset, fo)

		p.release()

	if cfg.use_high_prob_test_sample:
		if not os.path.isfile(cfg.high_prob_test_preprocessed_file):
			logger.info('Preprocessing high prob test data file...')
			p = PreProcessor(cfg)
			raw_data_path = cfg.high_prob_test_file
			croups, flag_croups, dataset = p.preprocess_dataset(raw_data_path)
			with open(cfg.high_prob_test_croups_file, 'w') as fo:
				json.dump(croups, fo)
			with open(cfg.high_prob_test_flag_croups_file, 'w') as fo:
				json.dump(flag_croups, fo)
			with open(cfg.high_prob_test_preprocessed_file, 'w') as fo:
				json.dump(dataset, fo)

			p.release()

def preprocess_test(cfg):
	logger = logging.getLogger("Military AI")
	logger.info('Preparing ' + cfg.cut_word_method + ' cut test data set...')
	logger.info('Checking raw test data files...')
	assert os.path.exists(cfg.test_raw_file), '{} file does not exist.'.format(cfg.test_raw_file)
	logger.info('Preparing the directories...')
	for dir_path in [cfg.vocab_dir, cfg.summary_dir]:
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	logger.info('Checking processed test data file...')
	if not os.path.isfile(cfg.test_preprocessed_file):
		logger.info('Preprocessing test data file...')
		p = PreProcessor(cfg)
		raw_data_path = cfg.test_raw_file
		croups, flag_croups, dataset = p.preprocess_dataset(raw_data_path)
		with open(cfg.test_croups_file, 'w') as fo:
			json.dump(croups, fo)
		with open(cfg.test_flag_croups_file, 'w') as fo:
			json.dump(flag_croups, fo)
		with open(cfg.test_preprocessed_file, 'w') as fo:
			json.dump(dataset, fo)

		p.release()


def prepare_vocab(cfg):
	logger = logging.getLogger("Military AI")
	if not os.path.isfile(cfg.token_embed_file):
		with open(cfg.train_croups_file, 'r') as fp:
			croups = json.load(fp)
		if cfg.use_test_vocab:
			with open(cfg.test_croups_file, 'r') as fp:
				croups += json.load(fp)
		if cfg.use_high_prob_test_sample:
			with open(cfg.high_prob_test_croups_file, 'r') as fp:
				croups += json.load(fp)

		logger.info("Training token embedding model")
		token_wv = Word2Vec(croups,
							size=cfg.token_emb_dim,
							window=5, compute_loss=True,
							min_count=cfg.token_min_cnt, iter=75,
							workers=multiprocessing.cpu_count()).wv
		token_wv.save(cfg.token_embed_file)
		del token_wv, croups

	if not os.path.isfile(cfg.flag_embed_file):
		with open(cfg.train_flag_croups_file, 'r') as fp:
			flag_croups = json.load(fp)
		if cfg.use_test_vocab:
			with open(cfg.test_flag_croups_file, 'r') as fp:
				flag_croups += json.load(fp)
		if cfg.use_high_prob_test_sample:
			with open(cfg.high_prob_test_flag_croups_file, 'r') as fp:
				flag_croups += json.load(fp)

		logger.info("Training flag embedding model")
		flag_wv = Word2Vec(flag_croups,
							size=cfg.flag_emb_dim,
							window=5, compute_loss=True,
							min_count=1, iter=75,
							workers=multiprocessing.cpu_count()).wv
		flag_wv.save(cfg.flag_embed_file)

	logger.info("Done!")


def prepare_data(cfg):


	preprocess_train(cfg)
	preprocess_test(cfg)
	prepare_vocab(cfg)


if __name__ == '__main__':
	logger = logging.getLogger("Military AI")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	for cfg in jieba_cfg, pyltp_cfg:
		prepare_data(cfg)