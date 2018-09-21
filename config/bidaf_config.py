# coding = utf-8
from config import base_config


class ConfigBidaf1(base_config.ConfigBase):
	# model
	algo = 'BIDAF'
	suffix = '_1'

	# model
	hidden_size = 200


class ConfigBidaf2(base_config.ConfigBase):
	# model
	algo = 'BIDAF'
	suffix = '_2'

	# model
	hidden_size = 100
	dropout_keep_prob = 0.9



class ConfigBidaf3(base_config.ConfigBase):
	# model
	algo = 'BIDAF'
	suffix = '_3'

	# model
	hidden_size = 100
	dropout_keep_prob = 0.7


config_1 = ConfigBidaf1()
config_2 = ConfigBidaf2()
config_3 = ConfigBidaf3()
