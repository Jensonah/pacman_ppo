

class __loss_calculator__():

	def __init__(self, gamma) -> None:

		self.gamma = gamma

		self.critic_values_best_t = None
		self.critic_values_action_t = None

		self.critic_values_best_t1 = None
		self.critic_values_action_t1 = None

		self.rewards = None

		self.retrieved_critic_loss = False
		self.retrieved_advantage = False

