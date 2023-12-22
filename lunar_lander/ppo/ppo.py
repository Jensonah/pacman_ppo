import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


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


def normalize_state(state):

	state[0] /= 1.5
	state[1] /= 1.5
	state[2] /= 5.0
	state[3] /= 5.0
	state[4] /= 3.1415927
	state[5] /= 5.0	
	# idx 6 and 7 are bools

	return state


def collect_episode(env, model):

	episode = []
	terminated = False
	truncated = False

	new_state, info = env.reset()
	
	while not (terminated or truncated):

		state = normalize_state(new_state)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		
		action, log_probs = model.act(state)

		new_state, reward, terminated, truncated, info = env.step(action)

		episode.append((state, action, log_probs, reward))

	return episode


def get_probs(actor, states, actions):
	all_probs = [actor.forward(state) for state in states]
	# get probabilities for actions in given state
	return [all_probs[i][actions[i]] for i in range(len(actions))]
	# index probability of action taken at sampling time


def update_net(gradients, optim, retain):
	
	policy_loss = torch.cat(gradients).sum()
	optim.zero_grad()
	policy_loss.backward(retain_graph=retain)
	optim.step()

	return policy_loss


def train(actor, critic, optim_actor, optim_critic, num_episodes, num_actors, num_epochs, eps = 0.2, gamma = 0.99):

	# One style of policy gradient implementation, popularized in [Mni+16] and well-suited for use
	# with recurrent neural networks, runs the policy for T timesteps (where T is much less than the
	# episode length), and uses the collected samples for an update. This style requires an advantage
	# estimator that does not look beyond timestep T .
	# Don't we already have this? Think about this. Or isn't that the entire idea of the estimator?

	obj_func_hist = []

	for i in range(num_episodes):

		episodes = [collect_episode(actor, critic) for _ in range(num_actors)]
		'''
		In other implementations the data is not a collection of varying length concluded episodes,
		but instead an array of i*j, where j is a fixed number of steps.
		So a row could be a partial episode (most likely), or multiple entire episodes, 
		with possibly the last one being truncated.
		'''

		states = [[state for state, _, _, _ in episode] for episode in episodes]
		actions = [[action for _, action, _, _ in episode] for episode in episodes]
		original_probs  = [[prob for _, _, prob, _ in episode] for episode in episodes]
		rewards = [[reward for _, _, _, reward in episode] for episode in episodes]

		for k in range(num_epochs):

			actor_gradient = []
			critic_values_t  = [critic.forward(state) for state in states]

			critic_values_t1 = critic_values_t.copy()
			critic_values_t1.pop(0)
			critic_values_t1.append(0) # in case of truncated episode this should be sampled from critic

			advantage = [gamma*critic_values_t1 + rewards - critic_values_t]

			for j in range(num_actors):

				# we need the probability for each action
				if k == 0:
					actor_probs = original_probs
				else:
					actor_probs = get_probs(actor, states[j], actions[j])

				difference_grad = actor_probs[j] / original_probs[j] # Make sure that this is element wise 
				# Note that for k = 0 these are all ones, but we keep the calculation such that the
				# backtracking algorithm can also see this

				advantage = [gamma*critic_values_t1 + rewards - critic_values_t] # again make sure element wise
				
				clipped = torch.clamp(difference_grad, 1 - eps, 1 + eps)
				ppo_gradient = torch.minimum(difference_grad*advantage, clipped*advantage)

				actor_gradient += ppo_gradient # Let's hope the shape remains correct here
			
			# we could also include an "entropy" bonus to the actor loss that encourages exploration
			critic_gradient = advantage*advantage # again make sure that this is element wise

			# update both models
			actor_loss = update_net(actor_gradient, optim_actor, retain=True)
			critic_loss = update_net(critic_gradient, optim_critic, retain=False)

		cum_rewards = np.array([sum(episode) for episode in rewards])
		obj_func_hist.append(cum_rewards.mean(), cum_rewards.min(), cum_rewards.max())

	return obj_func_hist

