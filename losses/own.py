import torch
from losses.ppo_q import PPO_Q_loss_calculator

class Own_loss_calculator(PPO_Q_loss_calculator):

	def __init__(self, gamma) -> None:
		super(Own_loss_calculator, self).__init__(gamma)


	def get_critic_loss(self):

		self.retrieved_critic_loss = self.safety_check(self.retrieved_critic_loss)

		constant = 1.25
		mse = constant *          (self.gamma*self.critic_values_best_t1 - (self.critic_values_action_t - self.rewards))**2
		mae = constant * torch.abs(self.gamma*self.critic_values_best_t1 - (self.critic_values_action_t - self.rewards))

		self.retrieved_critic_loss = True

		return torch.mean(torch.maximum(mse, mae))