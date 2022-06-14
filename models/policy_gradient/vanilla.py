import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
from gym.spaces import Discrete, box
from ..interfaces import Agent

class VanillaPG(Agent):
	def __init__(self, env, config):
		super().__init__(env, config)
		self.__init_policy()
		self.batch_obs = []
		self.batch_acts = []
		self.batch_weights = []
		self.batch_rets = []
		self.batch_lens = []
		self.ep_rewards = []
		self.lr = config["step_size"]
		self.gamma = config["gamma"]
		self.batch_size = config["batch_size"]
		self.optim = Adam(self.policy.parameters(), lr=self.lr)

	def __init_policy(self):
		layers = self.config["policy"]["layers"]
		in_size = self.env.observation_space.shape[0]
		out_size = self.env.action_space.n

		layer = lambda in_s, out_s: (nn.Linear(in_s, out_s), torch.nn.ReLU())
		mid_layers = [layer(in_size, layers[0]), *[layer(in_s, out_s) for in_s, out_s in zip(layers, layers[1:])]]
		mid_layers_unpacked = []
		for layer_tup in mid_layers:
			mid_layers_unpacked.append(layer_tup[0])
			mid_layers_unpacked.append(layer_tup[1])

		self.policy = nn.Sequential(
			*mid_layers_unpacked,
			nn.Linear(layers[-1], out_size)
		)

	def get_policy(self, observation):
		logits = self.policy(observation)
		return Categorical(logits=logits)

	def get_action(self, observation):
		return self.get_policy(observation).sample().item()

	def compute_loss(self, observation, action, weights):
		logp = self.get_policy(observation).log_prob(action)
		return -(logp * weights).mean()

	def update(self, observation, action, reward, next_observation, done):
		self.batch_obs.append(observation)
		self.batch_acts.append(action)
		self.ep_rewards.append(reward)

		if done:
			self.__update_policy()
			self.ep_rewards = []


	def __update_policy(self):
		ep_ret = sum(self.ep_rewards)
		ep_len = len(self.ep_rewards)
		self.batch_rets.append(ep_ret)
		self.batch_lens.append(ep_len)

		self.batch_weights += [ep_ret] * ep_len

		if len(self.batch_obs) > self.batch_size:
			self.optim.zero_grad()
			loss = self.compute_loss(
				torch.stack(self.batch_obs), 
				torch.as_tensor(self.batch_acts, dtype=torch.int),
				torch.as_tensor(self.batch_weights, dtype=torch.float)
				)
			loss.backward()
			self.optim.step()
			self.reset()

	def reset(self):
		self.batch_obs = []
		self.batch_acts = []
		self.batch_weights = []
		self.batch_rets = []
		self.batch_lens = []
		self.ep_rewards = []

	def soft_reset(self):
		self.batch_weights = []
		

	
