# coding = utf-8
class ConfigBase:
	# gpu
	gpu = '0'

	# dataset
	char_min_cnt = 1
	token_min_cnt = 3
	dev_split = 0.1
	seed = 502
	qtype_count = 10

	ans_max_token_len = 200
	ans_len_inter_size = 5

	# learning
	optim = 'adadelta'
	learning_rate = 0.5
	lr_decay = 0.95
	weight_decay = 0.0
	batch_size = 32
	epochs = 15
	loss_type = 'pointer'

	# model base
	dropout_keep_prob = 0.8
	embed_size = 300
	hidden_size = 150
	use_char_emb = False
	use_embe = True

	# paths
	root = './'

	train_raw_files = [root + 'data/train/question.json']
	train_preprocessed_files = [root + 'data/train/question_preprocessed.json']

	test_raw_files = []
	test_preprocessed_files = []

	char_embed_file = root + 'data/embedding/char_embed75.wv'
	token_embed_file = root + 'data/embedding/token_embed300.wv'
	elmo_dict_file = root + 'data/embedding/elmo-military_vocab.txt'
	elmo_embed_file = root + 'data/embedding/elmo-military_emb.pkl'

	data_dir = root + 'data/train'
	vocab_dir = root + 'data/embedding'
	model_dir = root + 'data/models/'
	result_dir = root + 'data/results'
	summary_dir = root + 'data/summary'
	log_path = None


config = ConfigBase()