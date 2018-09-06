# coding = utf-8
from config import base_config


class ConfigMatchLSTM1(base_config.ConfigBase):
	# model
	algo = 'MLSTM'
	suffix = '_1'

	# model
	hidden_size = 150

	# loss_type = 'mrl_mix'
	loss_type = 'pointer'


class ConfigMatchLSTM2(base_config.ConfigBase):
	# model
	algo = 'MLSTM'
	suffix = '_2'

	# model
	hidden_size = 100

	# loss_type = 'mrl_mix'
	loss_type = 'pointer'


config_1 = ConfigMatchLSTM1()
config_2 = ConfigMatchLSTM2()