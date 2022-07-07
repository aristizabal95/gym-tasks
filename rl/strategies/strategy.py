from abc import ABC, abstractmethod

class Strategy(ABC):
	@abstractmethod
	def __init__(self, config: dict):
		"""Initializes an Exploration Strategy

		Args:
			config (dict): contents for configuring the exploration strategy
		"""

	@abstractmethod
	def get_action(self, observation, max_ac):
		"""Retrieves an action following an exploration method

		Args:
			observation (any): current observation of the environment
		"""

	@abstractmethod
	def get_dist(self, observation):
		"""Returns the action distribution over the observation

		Args:
			observation (any): Description of the observation from the environment
		"""

	@abstractmethod
	def update(self, observation):
		"""Updates the exploration method according to the current observation

		Args:
			observation (any): current observation of the environment
		"""
