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
	def act(self, observation: np.ndarray):
		"""
		Abstract method for agents to implement.
		"""
		pass