class DataSetConfigBase:

	flag_emb_dim = 100
	token_emb_dim = 300
	token_min_cnt = 3
	dev_split = 0.2
	cv = 1
	seed = 502

	article_sample_len = 500

	# paths
	root = './'

	use_jieba = False
	use_test_vocab = False

	train_raw_file = root + 'data/train/question.json'
	test_raw_file = root + 'data/test/train-test-question.json'

	elmo_dict_file = root + 'data/embedding/elmo-military_vocab.txt'
	elmo_embed_file = root + 'data/embedding/elmo-military_emb.pkl'

	data_dir = root + 'data/train'
	vocab_dir = root + 'data/embedding'

	summary_dir = root + 'data/summary'
	log_path = None





config = DataSetConfigBase()
