import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .. import Strategy

class EpsilonGreedy(Strategy):
	def __init__(self, config: dict):
		self.config = config
		self.epsilon = config["epsilon"]
		self.lr = config["lr"]
		self.env = config["env"]
		self.weight_decay = config["weight_decay"]
		self.num_actions = self.env.action_space.n
		self.__init_action_value()
		self.optim = Adam(self.action_value.parameters(), lr=self.lr, weight_decay=self.weight_decay)
		self.seed = config["seed"]
		self.rand = np.random.default_rng(self.seed)

	def __call__(self, observation):
		return self.action_value(observation)

	def __init_action_value(self):
		layers = self.config["action_value"]["layers"]
		in_size = self.env.observation_space.shape[0]
		out_size = self.num_actions

		layer = lambda in_s, out_s: (nn.Linear(in_s, out_s), torch.nn.ReLU())
		mid_layers = [layer(in_size, layers[0]), *[layer(in_s, out_s) for in_s, out_s in zip(layers, layers[1:])]]
		mid_layers_unpacked = []
		for layer_tup in mid_layers:
			mid_layers_unpacked.append(layer_tup[0])
			mid_layers_unpacked.append(layer_tup[1])

		self.action_value = nn.Sequential(
			*mid_layers_unpacked,
			nn.Linear(layers[-1], out_size)
		)

	def get_action(self, observation):
		# Implement epsilon-greedy
		if self.rand.random() < self.epsilon:
			action = self.rand.choice(self.num_actions)
		else:
			action = torch.argmax(self(observation)).item()

		return action

	def get_dist(self, observation):
		optimal_action = torch.argmax(self(observation))
		dist = np.array([self.epsilon/self.num_actions,]*self.num_actions)
		dist[optimal_action] += 1 - self.epsilon
		return dist

	def update(self, error):
		self.action_value.zero_grad()
		td_squared = error ** 2
		td_squared.backward()
		self.optim.step()
