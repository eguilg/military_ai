class DataSetConfigBase:

	def __init__(self, _cv=None, _use_jieba=None):
		if _cv:
			self.cv = _cv
		if _use_jieba:
			self.use_jieba = _use_jieba

	flag_emb_dim = 100
	token_emb_dim = 300
	token_min_cnt = 3
	dev_split = 0.2
	cv = 0
	# seed = 502 # for 5 folds
	seed = 10086

	article_sample_len = 500
	article_sample_len_test = 500

	# paths
	root = './'

	use_jieba = True
	use_test_vocab = False

	# train_raw_file = root + 'data/train/question.json'
	train_raw_file = root + 'data/train/high_prob_test_articles.txt'

	test_raw_file = root + 'data/test/test-question-cleaned.json'

	elmo_dict_file = root + 'data/embedding/elmo-military_vocab.txt'
	elmo_embed_file = root + 'data/embedding/elmo-military_emb.pkl'

	data_dir = root + 'data/train'
	vocab_dir = root + 'data/embedding'

	summary_dir = root + 'data/summary'
	log_path = None





config = DataSetConfigBase()
