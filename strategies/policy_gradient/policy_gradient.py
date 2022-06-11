import torch
import copy
from ..interfaces import Strategy

class PolicyGradient(Strategy):
	"""
	Neural network strategy.
	"""
	def __init__(self, observation_space, action_space, config):
		super().__init__(observation_space, action_space, config)
		self.__init_model()
		self.model_copy = copy.deepcopy(self.model)
		self.memory = []
		self.batch_size = config["batch_size"]
		self.memory_size = config["memory_size"]

	def __init_model(self):
		layers = self.config["layers"]
		in_size = self.observation_space.shape[0]
		out_size = self.config["out_size"]

		layer = lambda in_s, out_s: (torch.nn.Linear(in_s, out_s), torch.nn.ReLU())
		mid_layers = [layer(in_size, layers[0]), *[layer(in_s, out_s) for in_s, out_s in zip(layers, layers[1:])]]
		mid_layers_unpacked = []
		for layer_tup in mid_layers:
			mid_layers_unpacked.append(layer_tup[0])
			mid_layers_unpacked.append(layer_tup[1])

		self.model = torch.nn.Sequential(
			*mid_layers_unpacked,
			torch.nn.Linear(layers[-1], out_size)
		)

	def forward(self, observation):
		return self.model(observation)

	def backward(self, observation, action, reward, next_observation, done):
		pi_t = self.model(observation)
		pi_t1 = self.model_copy(next_observation)
		pass
