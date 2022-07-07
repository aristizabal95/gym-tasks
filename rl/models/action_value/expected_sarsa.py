
import torch
import numpy as np
from gym.spaces import Discrete, box
from ..interfaces import Agent
from ...strategies import Strategy

class ExpectedSarsa(Agent):
	def __init__(self, env, config: dict, strategy: Strategy):
		super().__init__(env, config)
		self.lr = config["step_size"]
		self.gamma = config["gamma"]
		self.strategy = strategy
		
	def get_action(self, observation):
		return self.strategy.get_action(observation)

	def update(self, observation, action, reward, next_observation, done):
		prev_val = self.strategy(observation)[action]
		with torch.no_grad():
			# we don't want to compute the gradients with respect to the future actions
			if done:
				expected_next_val = 0
			else:
				n_val = self.strategy(next_observation)
				dist = self.strategy.get_dist(next_observation)
				expected_next_val = np.dot(n_val, dist)
		
		# negative as we want to do gradient ascent
		error = (reward + self.gamma*expected_next_val - prev_val)
		self.strategy.update(error)
		self.reset()


	def reset(self):
		pass

	def soft_reset(self):
		pass