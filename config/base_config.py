# coding = utf-8
class ConfigBase:
	# gpu
	gpu = '0'

	test = True

	# learning
	optim = 'adadelta'
	learning_rate = 0.5
	lr_decay = 0.95
	weight_decay = 0.0
	batch_size = 32
	epochs = 15
	loss_type = 'pointer'
	qtype_count = 10
	# model base
	dropout_keep_prob = 0.8
	embed_size = 300
	hidden_size = 150
	use_char_emb = False
	use_embe = True

	root = './'
	model_dir = root + 'data/models/'
	result_dir = root + 'data/results'
config = ConfigBase()
