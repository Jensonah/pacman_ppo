import torch
from losses.base import __loss_calculator__

class PPO_Q_loss_calculator(__loss_calculator__):

	def __init__(self, gamma) -> None:
		super(PPO_Q_loss_calculator, self).__init__(gamma)

	
	def update_losses(self, critic_values, actions, rewards):

		len_episode = len(rewards)
		
		self.critic_values_best_t = torch.max(critic_values, 1).values.unsqueeze(1)
		self.critic_values_best_t1 = torch.cat((self.critic_values_best_t.clone()[1:], torch.zeros(1,1)))
		
		# Here we index for each state the value of that state paired with the action taken
		self.critic_values_action_t = critic_values[torch.arange(len_episode), actions].unsqueeze(1)
		self.critic_values_action_t1 = torch.cat((self.critic_values_action_t.clone()[1:], torch.zeros(1,1)))

		self.rewards = rewards

		self.retrieved_critic_loss = False
		self.retrieved_advantage = False


	def get_critic_loss(self):

		if self.retrieved_critic_loss:
			raise RuntimeError("""Losses should first be updated using update_losses() before retrieval.
						Make sure to not call this function twice in a row.""")
		
		self.retrieved_critic_loss = True
		
		return torch.mean( (self.gamma*self.critic_values_best_t1 - (self.critic_values_action_t - self.rewards))**2 )


	def get_advantage(self):

		if self.retrieved_advantage:
			raise RuntimeError("""Losses should first be updated using update_losses() before retrieval.
						Make sure to not call this function twice in a row.""")
		
		self.retrieved_advantage = True

		# Here we are comparing two state value estimation, one made one timestep later than the other
        # Their difference is zero if in that timestep we took the optimal action
		return self.gamma*self.critic_values_action_t1 - (self.critic_values_action_t - self.rewards)

