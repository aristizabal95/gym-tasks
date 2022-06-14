from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
	"""
	Abstract class for agents.
	"""
	@abstractmethod
	def __init__(self, env, config):
		self.env = env
		self.config = config
	
	@abstractmethod
	def get_action(self, observation: np.ndarray):
		"""
		Abstract method for agents to implement.
		"""
		pass

	@abstractmethod
	def update(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool):
		"""
		Abstract method for agents to implement.
		"""
		pass