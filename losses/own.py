import torch
from ppo_q import PPO_Q_loss_calculator

class Own_loss_calculator(PPO_Q_loss_calculator):

	def __init__(self, gamma) -> None:
		super(Own_loss_calculator, self).__init__(gamma)

	def get_critic_loss(self):
		raise NotImplementedError()