from abc import ABC, abstractmethod

class Strategy(ABC):
	"""
	Abstract class for strategies.
	"""
	@abstractmethod
	def __init__(self, observation_space, action_space, config):
		self.observation_space = observation_space
		self.action_space = action_space
		self.config = config

	@abstractmethod
	def forward(self, observation):
		"""
		Compute current observation value
		"""
		pass

	@abstractmethod
	def backward(self, observation, action, reward, next_observation, done):
		"""
		Update the strategy's parameters.
		"""
		pass