

class LossFactory():

	def create_loss_calculator(loss_type, gamma):

		if loss_type == "ppo":
			from losses.ppo import PPO_loss_calculator
			loss_calculator = PPO_loss_calculator(gamma)
		
		elif loss_type == "ppo_q":
			from losses.ppo_q import PPO_Q_loss_calculator
			loss_calculator = PPO_Q_loss_calculator(gamma)

		elif loss_type == "own":
			from losses.own import Own_loss_calculator
			loss_calculator = Own_loss_calculator(gamma)
		
		else:
			raise NotImplementedError(f"{loss_type} not part of functionality")

		return loss_calculator