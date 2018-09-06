# coding = utf-8
"""
This module prepares and runs the whole system.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import logging
from dataset import MilitaryAiDataset
from rc_model import RCModel
from config import base_config, bidaf_config, mlstm_config

base_cfg = base_config.config
bidaf_1_cfg = bidaf_config.config_1
bidaf_2_cfg = bidaf_config.config_2
mlstm_1_cfg = mlstm_config.config_1
mlstm_2_cfg = mlstm_config.config_2

# setting current config
# cur_cfg = bidaf_1_cfg
# cur_cfg = bidaf_2_cfg
cur_cfg = mlstm_1_cfg
# cur_cfg = mlstm_2_cfg

def parse_args(cfg):
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
	model_settings = parser.add_argument_group('model settings')
	model_settings.add_argument('--suffix', type=str, default=cfg.suffix,
								help='model file name suffix')
	model_settings.add_argument('--is_restore', type=int, default=0, help='is restore model from file')
	model_settings.add_argument('--restore_suffix', type=str, default=None,
								help='restore model from file')

	return parser.parse_args()


def prepare(args, cfg):
	"""
	checks data, creates the directories, prepare the vocabulary and embeddings
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Checking the data files...')
	for data_path in cfg.train_raw_files:
		assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
	logger.info('Preparing the directories...')
	for dir_path in [cfg.vocab_dir, cfg.model_dir, cfg.result_dir, cfg.summary_dir]:
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

	logger.info('Building vocabulary...')
	logger.info('Load data_set and vocab...')

	mai_data = MilitaryAiDataset(cfg)

	logger.info('Assigning embeddings...')
	if not cfg.use_embe:
		mai_data.token_vocab.randomly_init_embeddings(cfg.embed_size)
		mai_data.char_vocab.randomly_init_embeddings(cfg.embed_size)

	logger.info('Done with preparing!')


def train(args, cfg):
	"""
	trains the reading comprehension model
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Load data_set and vocab...')

	mai_data = MilitaryAiDataset(cfg)

	logger.info('Assigning embeddings...')
	if not cfg.use_embe:
		mai_data.token_vocab.randomly_init_embeddings(cfg.embed_size)
		mai_data.char_vocab.randomly_init_embeddings(cfg.embed_size)
	logger.info('Initialize the model...')
	rc_model = RCModel(mai_data.char_vocab, mai_data.token_vocab,
					   mai_data.flag_vocab, mai_data.elmo_vocab, cfg)
	if args.is_restore or args.restore_suffix:
		restore_prefix = cfg.algo + args.suffix
		if args.restore_suffix:
			restore_prefix = cfg.algo + args.restore_suffix
		rc_model.restore(model_dir=cfg.model_dir, model_prefix=restore_prefix)
	logger.info('Training the model...')
	rc_model.train(mai_data, cfg.epochs, cfg.batch_size, save_dir=cfg.model_dir,
				   save_prefix=cfg.algo + args.suffix,
				   dropout_keep_prob=cfg.dropout_keep_prob)
	logger.info('Done with model training!')


def evaluate(args, cfg):
	"""
	evaluate the trained model on dev files
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Load data_set and vocab...')
	mai_data = MilitaryAiDataset(cfg)

	logger.info('Assigning embeddings...')
	if not cfg.use_embe:
		mai_data.token_vocab.randomly_init_embeddings(cfg.embed_size)
		mai_data.char_vocab.randomly_init_embeddings(cfg.embed_size)
	logger.info('Restoring the model...')
	rc_model = RCModel(mai_data.char_vocab, mai_data.token_vocab,
					   mai_data.flag_vocab, mai_data.elmo_vocab, cfg)
	rc_model.restore(model_dir=cfg.model_dir, model_prefix=cfg.algo + args.suffix)
	logger.info('Evaluating the model on dev set...')
	dev_batches = mai_data.gen_mini_batches('dev', cfg.batch_size, shuffle=False)
	dev_loss, dev_main_loss, dev_bleu_rouge = rc_model.evaluate(
		dev_batches, result_dir=cfg.result_dir, result_prefix='dev.predicted')
	logger.info('Loss on dev set: {}'.format(dev_main_loss))
	logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
	logger.info('Predicted answers are saved to {}'.format(os.path.join(cfg.result_dir)))


def predict(args, cfg):
	"""
	predicts answers for test files
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Load data_set and vocab...')
	mai_data = MilitaryAiDataset(cfg)
	logger.info('Assigning embeddings...')
	if not cfg.use_embe:
		mai_data.token_vocab.randomly_init_embeddings(cfg.embed_size)
		mai_data.char_vocab.randomly_init_embeddings(cfg.embed_size)
	logger.info('Restoring the model...')
	rc_model = RCModel(mai_data.char_vocab, mai_data.token_vocab,
					   mai_data.flag_vocab, mai_data.elmo_vocab, cfg)
	rc_model.restore(model_dir=cfg.model_dir, model_prefix=cfg.algo + args.suffix)
	logger.info('Predicting answers for test set...')
	test_batches = mai_data.gen_mini_batches('test', cfg.batch_size, shuffle=False)
	rc_model.evaluate(test_batches,
					  result_dir=cfg.result_dir, result_prefix='test.predicted')


def run(cfg):
	"""
	Prepares and runs the whole system.
	"""
	args = parse_args(cfg)

	logger = logging.getLogger("Military AI")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	if cfg.log_path:
		file_handler = logging.FileHandler(cfg.log_path)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	else:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)
		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)

	logger.info('Running with args : {}'.format(args))
	logger.info('Running with model cfg : {}'.format(cfg.__class__.__dict__))
	logger.info('Running with base cfg : {}\n'.format(cfg.__class__.__bases__[0].__dict__))

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

	if args.prepare:
		prepare(args, cfg)
	if args.train:
		train(args, cfg)
	if args.evaluate:
		evaluate(args, cfg)
	if args.predict:
		predict(args, cfg)


if __name__ == '__main__':

	run(cur_cfg)
