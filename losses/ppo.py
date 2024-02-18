import torch

class PPO_loss_calculator():

	def __init__(self, gamma) -> None:

		self.gamma = gamma

		self.critic_values_t = None
		self.critic_values_t1 = None

		self.rewards = None

		self.retrieved_critic_loss = False
		self.retrieved_advantage = False

	
	def update_losses(self, critic_values, _, rewards):
		
		self.critic_values_t = critic_values
		self.critic_values_t1 = torch.cat((critic_values.clone()[1:], torch.zeros(1,1)))

		self.rewards = rewards

		self.retrieved_critic_loss = False
		self.retrieved_advantage = False

	
	def safety_check(self, already_retrieved):

		if already_retrieved:
			raise RuntimeError("""Losses should be updated using update_losses() before retrieval.
						Make sure to not call this function twice in a row.""")
		
		return True


	def get_critic_loss(self):

		self.retrieved_critic_loss = self.safety_check(self.retrieved_critic_loss)
		
		return torch.mean( (self.gamma*self.critic_values_t1 - (self.critic_values_t - self.rewards))**2 )


	def get_advantage(self):
		
		self.retrieved_advantage = self.safety_check(self.retrieved_advantage)

		# Here we are comparing two state value estimation, one made one timestep later than the other
        # Their difference is zero if in that timestep we took the optimal action
		return self.gamma*self.critic_values_t1 - (self.critic_values_t - self.rewards)
