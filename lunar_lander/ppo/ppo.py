import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Actor(nn.Module):

	def __init__(self,	):
		super(Actor, self).__init__()
		# define layers here
		

	def forward(self, x):

		x = ...
		
		return x
	

	def act(self, state):

		probabilities = self.forward(state)
		probs = Categorical(probabilities)
		action = probs.sample()
		return action.item(), probs.log_prob(action)
	

class Critic(nn.Module):

	def __init__(self,	):
		super(Critic, self).__init__()
		# define layers here
		

	def forward(self, x):

		x = ...

		return x



# for iteration=1, 2, . . . do
# 	for actor=1, 2, . . . , N do
# 		Run policy πθold in environment for T timesteps
# 		Compute advantage estimates Â1 , . . . , ÂT
# 	end for
# 	Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ N T
# 	θold ← θ
# end for


def collect_episode(env, model):

	episode = []

	# the library technically returns a state observation, but in 
	# our case the state and the state observation are equal
	state, info = env.reset(seed=42) # TODO: remove seed
	state = normalize_state(state)
	state = torch.from_numpy(state).float().unsqueeze(0).to(device)

	terminated = False
	truncated = False

	# the library should already include penalties for terminated, for truncated there is no need
	while not (terminated or truncated):
		
		action, log_probs = model.act(state)

		state, reward, terminated, truncated, info = env.step(action)
		state = normalize_state(state)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)

		# TODO: Maybe add truncated here
		episode.append((state, log_probs, reward))

	return episode


def train(actor, critic, num_episodes, num_actors):

	for i in range(num_episodes):

		for j in range(num_actors):
			episode = collect_episode(actor, critic)
			'''
			In other implementations the data is not a collection of varying length episodes,
			but instead an array of i*j, where j is a fixed number of steps.
			So a row could be a partial episode (most likely), or multiple entire episodes, 
			with possibly the last one being truncated.
			'''

		# train on data