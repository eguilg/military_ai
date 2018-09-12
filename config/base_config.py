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
	qtype_count = 10

	loss_type = 'pointer'
	# loss_type = 'mrl_mix'
	# loss_type = 'mrl_soft'
	# loss_type = 'mrl_hard'

	switch = True

	# model base
	dropout_keep_prob = 0.8
	embed_size = 300
	hidden_size = 150


	root = './'
	model_dir = root + 'data/models/'
	result_dir = root + 'data/results'
config = ConfigBase()
