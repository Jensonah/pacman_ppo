

class LossFactory():

	def create_loss_calculator(loss_type, gamma):

		if loss_type == "ppo":
			raise NotImplementedError("PPO loss not implemented")
		
		elif loss_type == "ppo_q":
			from losses.ppo_q import PPO_Q_loss_calculator
			loss_calculator = PPO_Q_loss_calculator(gamma)

		elif loss_type == "own":
			raise NotImplementedError("Own loss not implemented")
		
		else:
			raise NotImplementedError(f"{loss_type} not part of functionality")

		return loss_calculator