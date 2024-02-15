

class ModelFactory():

	def create_model(env_name, device, mode):
		
		if env_name == "LunarLander-v2":
			from models.lander_models import Actor, Critic

		elif env_name == "ALE/MsPacman-v5":
			from models.pacman_models import Actor, Critic
			
		else:
			raise NotImplementedError("Model not implemented for this environment")
		
		return Actor(device, mode), Critic(device, mode)