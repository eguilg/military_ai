from config.dataset_base_config import DataSetConfigBase


class DataSetConfigJieba(DataSetConfigBase):

	cut_word_method = 'jieba'

	train_preprocessed_file = DataSetConfigBase.root + 'data/train/train_preprocessed_' + cut_word_method + '_' + str(
		DataSetConfigBase.article_sample_len) + '.json'

	train_croups_file = DataSetConfigBase.root + 'data/train/train_croups_' + cut_word_method + '.json'

	train_flag_croups_file = DataSetConfigBase.root + 'data/train/train_flag_croups_' + cut_word_method + '.json'

	test_preprocessed_file = DataSetConfigBase.root + 'data/test/test_preprocessed_' + cut_word_method + '_' + str(
		DataSetConfigBase.article_sample_len_test) + '.json'

	test_croups_file = DataSetConfigBase.root + 'data/test/test_croups_' + cut_word_method + '.json'
	test_flag_croups_file = DataSetConfigBase.root + 'data/test/test_flag_croups_' + cut_word_method + '.json'

	flag_embed_file = DataSetConfigBase.root + 'data/embedding/flag_embed' + str(
		DataSetConfigBase.flag_emb_dim) + '_' + cut_word_method + '.wv'
	token_embed_file = DataSetConfigBase.root + 'data/embedding/token_embed' + str(
		DataSetConfigBase.token_emb_dim) + '_' + cut_word_method + '.wv'

	# extra
	jieba_big_dict_path = './data/embedding/dict.txt.big'


config = DataSetConfigJieba()
