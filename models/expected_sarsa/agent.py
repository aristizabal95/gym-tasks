from ..interfaces import Agent
import numpy as np

class ExpectedSarsa(Agent):
	"""
	Expected Sarsa agent.
	"""
	def __init__(self, env, config):
		super().__init__(env, config)

	def act(self, observation: np.ndarray):
		"""
		Implement the agent's policy.
		"""
		action = self.env.action_space.sample()	
		return action