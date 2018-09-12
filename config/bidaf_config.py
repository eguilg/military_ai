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
	hidden_size = 150




config_1 = ConfigBidaf1()
config_2 = ConfigBidaf2()
