import torch
import copy
from ...interfaces import Strategy

class ActorCritic(Strategy):
	"""
	Actor Critic computation strategy.
	"""
	def __init__(self, observation_space, action_space, config):
		super().__init__(observation_space, action_space, config)
		self.__init_policy()
		self.__init_value()
		self.memory = []
		self.batch_size = config["batch_size"]
		self.memory_size = config["memory_size"]
		self.gamma = config["gamma"]

	def __init_policy(self):
		layers = self.config["policy"]["layers"]
		in_size = self.observation_space.shape[0]
		out_size = self.action_space.n

		layer = lambda in_s, out_s: (torch.nn.Linear(in_s, out_s), torch.nn.ReLU())
		mid_layers = [layer(in_size, layers[0]), *[layer(in_s, out_s) for in_s, out_s in zip(layers, layers[1:])]]
		mid_layers_unpacked = []
		for layer_tup in mid_layers:
			mid_layers_unpacked.append(layer_tup[0])
			mid_layers_unpacked.append(layer_tup[1])

		self.policy = torch.nn.Sequential(
			*mid_layers_unpacked,
			torch.nn.Linear(layers[-1], out_size)
		)

	def __init_value(self):
		layers = self.config["value"]["layers"]
		in_size = self.observation_space.shape[0]
		out_size = 1

		layer = lambda in_s, out_s: (torch.nn.Linear(in_s, out_s), torch.nn.ReLU())
		mid_layers = [layer(in_size, layers[0]), *[layer(in_s, out_s) for in_s, out_s in zip(layers, layers[1:])]]
		mid_layers_unpacked = []
		for layer_tup in mid_layers:
			mid_layers_unpacked.append(layer_tup[0])
			mid_layers_unpacked.append(layer_tup[1])

		self.value = torch.nn.Sequential(
			*mid_layers_unpacked,
			torch.nn.Linear(layers[-1], out_size)
		)


	def forward(self, observation):
		return self.model(observation)

	def backward(self, observation, action, reward, next_observation, done):
		cur_val = self.value(observation)
		next_val = self.value(next_observation)
		value_loss = reward + self.gamma * next_val - cur_val
		pi_t = self.model(observation)
		pass

	def reset(self):
		self.__init_model()