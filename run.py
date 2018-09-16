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
from prepare import prepare_data
from config import base_config, bidaf_config, mlstm_config, dataset_base_config, jieba_data_config, pyltp_data_config

base_cfg = base_config.config
bidaf_1_cfg = bidaf_config.config_1
bidaf_2_cfg = bidaf_config.config_2
mlstm_1_cfg = mlstm_config.config_1
mlstm_2_cfg = mlstm_config.config_2

data_base_cfg = dataset_base_config.config
jieba_cfg = jieba_data_config.config
pyltp_cfg = pyltp_data_config.config

# setting current config
cur_model_cfg = bidaf_1_cfg


# cur_cfg = bidaf_2_cfg
# cur_cfg = mlstm_1_cfg
# cur_cfg = mlstm_2_cfg


def parse_args(model_cfg):
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
	model_settings.add_argument('--suffix', type=str, default=model_cfg.suffix,
								help='model file name suffix')
	model_settings.add_argument('--is_restore', type=int, default=0, help='is restore model from file')
	model_settings.add_argument('--restore_suffix', type=str, default=None,
								help='restore model from file')

	return parser.parse_args()


def prepare(args, pyltp_cfg, jieba_cfg):
	"""
	checks data, creates the directories, prepare the vocabulary and embeddings
	"""
	logger = logging.getLogger("Military AI")

	prepare_data(pyltp_cfg)
	prepare_data(jieba_cfg)

	logger.info('Done with preparing!')


def train(args, pylpt_cfg, jieba_cfg, model_cfg):
	"""
	trains the reading comprehension model
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Load data set and vocab...')

	mai_data = MilitaryAiDataset(pylpt_cfg, jieba_cfg)

	logger.info('Initialize the model...')
	rc_model = RCModel(mai_data, model_cfg)

	if args.is_restore or args.restore_suffix:
		restore_prefix = 'pointer_pyltp_best'
		if args.restore_suffix:
			restore_prefix = args.restore_suffix
		rc_model.restore(model_dir=model_cfg.model_dir, model_prefix=restore_prefix)
	logger.info('Training the model...')
	rc_model.train(model_cfg.epochs, model_cfg.batch_size, save_dir=model_cfg.model_dir,
				   dropout_keep_prob=model_cfg.dropout_keep_prob)
	logger.info('Done with model training!')


def evaluate(args, pyltp_cfg, jieba_cfg, model_cfg):
	"""
	evaluate the trained model on dev files
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Load data set and vocab...')
	mai_data = MilitaryAiDataset(pyltp_cfg, jieba_cfg, use_jieba=pyltp_cfg.use_jieba)

	if args.restore_suffix:
		restore_prefix = args.restore_suffix
	elif pyltp_cfg.use_jieba:
		restore_prefix = 'mrl_soft_jieba_best'
	else:
		restore_prefix = 'mrl_soft_pyltp_best'

	logger.info('Restoring the model...')
	rc_model = RCModel(mai_data, model_cfg)

	rc_model.restore(model_dir=model_cfg.model_dir,
					 model_prefix=restore_prefix)
	logger.info('Evaluating the model on dev set...')
	dev_batches = mai_data.gen_mini_batches('dev', model_cfg.batch_size, shuffle=False)
	dev_loss, dev_main_loss, dev_bleu_rouge = rc_model.evaluate(
		dev_batches, result_dir=model_cfg.result_dir,
		result_prefix=restore_prefix)
	logger.info('Loss on dev set: {}'.format(dev_main_loss))
	logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
	logger.info('Predicted answers are saved to {}'.format(os.path.join(model_cfg.result_dir)))


def predict(args, pyltp_cfg, jieba_cfg, model_cfg):
	"""
	predicts answers for test files
	"""
	logger = logging.getLogger("Military AI")
	logger.info('Load data set and vocab...')
	mai_data = MilitaryAiDataset(pyltp_cfg, jieba_cfg, test=True, use_jieba=pyltp_cfg.use_jieba)

	if args.restore_suffix:
		restore_prefix = args.restore_suffix
	elif pyltp_cfg.use_jieba:
		restore_prefix = 'mrl_soft_jieba_best'
	else:
		restore_prefix = 'mrl_soft_pyltp_best'

	logger.info('Restoring the model...')
	rc_model = RCModel(mai_data, model_cfg)
	rc_model.restore(model_dir=model_cfg.model_dir,
					 model_prefix=restore_prefix)
	logger.info('Predicting answers for test set...')
	test_batches = mai_data.gen_mini_batches('test', model_cfg.batch_size, shuffle=False)
	rc_model.predict_for_ensemble(test_batches,
								  result_dir=model_cfg.result_dir,
								  result_prefix=restore_prefix)


def run(pylpt_cfg, jieba_cfg, model_cfg):
	"""
	Prepares and runs the whole system.
	"""
	args = parse_args(model_cfg)

	logger = logging.getLogger("Military AI")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	if pylpt_cfg.log_path:
		file_handler = logging.FileHandler(pylpt_cfg.log_path)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	else:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)
		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)

	logger.info('Running with args : {}'.format(args))
	logger.info('Running with model cfg : {}'.format(model_cfg.__class__.__dict__))
	logger.info('Running with base cfg : {}\n'.format(model_cfg.__class__.__bases__[0].__dict__))

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = model_cfg.gpu

	if args.prepare:
		prepare(args, pyltp_cfg, jieba_cfg)
	if args.train:
		train(args, pyltp_cfg, jieba_cfg, model_cfg)
	if args.evaluate:
		evaluate(args, pyltp_cfg, jieba_cfg, model_cfg)
	if args.predict:
		predict(args, pyltp_cfg, jieba_cfg, model_cfg)


if __name__ == '__main__':
	run(pyltp_cfg, jieba_cfg, cur_model_cfg)
