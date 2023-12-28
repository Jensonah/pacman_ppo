import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
import time
import json


from base import Actor
from base import Critic
from base import collect_episode


# index probability of action taken at sampling time with current updated model
def get_probs(actor, states, actions):
	all_probs = torch.cat([actor.forward(state) for state in states])
	actions = torch.Tensor(actions).int()
	out = all_probs[torch.arange(all_probs.size(0)), actions]
	# these lines do this below more efficient
	#[all_probs[i][actions[i]].unsqueeze(0) for i in range(len(actions))]
	return out


def update_net(gradients, optim):
	
	policy_loss = gradients.sum()
	optim.zero_grad()
	policy_loss.backward()
	optim.step()

	return policy_loss


def get_standardized_tensor(xs):

	## eps is the smallest representable float, which is 
	# added to the standard deviation of the returns to avoid numerical instabilities   
	eps = np.finfo(np.float32).eps.item()     
	xs = torch.tensor(xs)
	return (xs - xs.mean()) / (xs.std() + eps)



# TODO: make sure the amount of context frames can be a variable 
# (such that a state consists of the last x frames)
def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, eps, gamma):

	# One style of policy gradient implementation, popularized in [Mni+16] and well-suited for use
	# with recurrent neural networks, runs the policy for T timesteps (where T is much less than the
	# episode length), and uses the collected samples for an update. This style requires an advantage
	# estimator that does not look beyond timestep T .
	# Don't we already have this? Think about this. Or isn't that the entire idea of the estimator?

	obj_func_hist = []
	losses = []

	time_sampling = 0
	time_updating = 0

	for _ in tqdm(range(num_episodes)):
		
		t0 = time.time()

		episodes = [collect_episode(env, actor) for _ in range(num_actors)]
		
		sum_tuples = 0
		for episode in episodes:
			sum_tuples += len(episode)

		'''
		In other implementations the data is not a collection of varying length concluded episodes,
		but instead an array of i*j, where j is a fixed number of steps.
		So a row could be a partial episode (most likely), or multiple entire episodes, 
		with possibly the last one being truncated.
		'''

		states = [[state for state, _, _, _ in episode] for episode in episodes]
		actions = [[action for _, action, _, _ in episode] for episode in episodes]
		original_probs  = [torch.cat([prob for _, _, prob, _ in episode]) for episode in episodes]
		rewards = [[reward for _, _, _, reward in episode] for episode in episodes]

		cum_rewards = np.array([sum(episode) for episode in rewards])
		obj_func_hist.append((cum_rewards.mean(), cum_rewards.min(), cum_rewards.max()))

		rewards = [get_standardized_tensor(reward).unsqueeze(1) for reward in rewards]

		t1 = time.time()

		time_sampling += t1 - t0

		t0 = time.time()
		for k in range(num_epochs):

			actor_gradient = torch.empty(0)
			critic_gradient = torch.empty(0)

			for j in range(num_actors):

				critic_values_t  = torch.cat([critic(state) for state in states[j]])
				critic_values_t1 = torch.cat((critic_values_t.clone()[1:], torch.zeros(1,1)))

				advantage = gamma*critic_values_t1 + rewards[j] - critic_values_t

				# we need the probability for each action
				if k == 0:
					actor_probs = original_probs[j].clone()
					original_probs[j] = original_probs[j].detach()
				else:
					# we can also always do this and detach already all above
					actor_probs = get_probs(actor, states[j], actions[j])

				difference_grad = torch.div(actor_probs, original_probs[j]).unsqueeze(1)
				# Note that for k = 0 these are all ones, but we keep the calculation such that the
				# backtracking algorithm can also see this
				
				clipped = torch.clamp(difference_grad, 1 - eps, 1 + eps)

				ppo_gradient = torch.minimum(difference_grad*advantage, clipped*advantage)
				ppo_gradient *= -1 # this seems to be the right place

				actor_gradient = torch.cat((actor_gradient, ppo_gradient))
				# we could also include an "entropy" bonus to the actor loss that encourages exploration
				
				critic_gradient = torch.cat((critic_gradient, advantage*advantage))

			# update both models
			gradient = torch.cat((actor_gradient, critic_gradient))
			loss = update_net(gradient, optim)
			losses.append(loss.detach())
		
		t1 = time.time()

		time_updating += t1 - t0

	print(f"Total time sampling = {time_sampling}")
	print(f"Total time updating = {time_updating}")

	return obj_func_hist, losses


def plot_fancy_loss(data, path):

	means, mins, maxes = list(map(list, zip(*data)))

	plt.clf()

	xs = range(len(data))

	# Plot points
	plt.plot(xs, means, 'b', alpha=0.8)
	#plt.plot(pfb_x_mean, pfb_y_mean, 'g', alpha=0.8)

	# Plot errors
	plt.fill_between(xs, mins, maxes, color='b', alpha=0.3)
	#plt.fill_between(pfb_x_mean, pfb_y_low, pfb_y_high, color='g', alpha=0.3)

	# Set labels
	plt.title(f'Reward projection')
	plt.xlabel('Episode No.')
	plt.ylabel(f'Reward')
	plt.legend(['My PPO implementation'])#, 'PPO for Beginners'])\
	plt.savefig(path)


def plot_ugly_loss(data, length, name):

	plt.clf()
	plt.plot(list(range(length)), data)
	plt.savefig(f"{base_path}/{name}loss.png")


pacman_hyperparameters = {
	"num_episodes" : 100,
	"gamma" : 0.99,
	"lr" : 1e-3,
	"env_name" : "ALE/MsPacman-v5",
	"frameskip" : 4,
	"repeat_action_probability": 0.2,
	"render_mode" : "rgb_array",
	"trial_number" : 2,
	"eps" : 0.2,
	"num_epochs" : 5,
	"num_actors" : 3,
	"device" : "cpu",
	"obs_type" : "rgb",
	"input_frame_dim" : (3,210,160)
}

# input (250, 160, 3)
# reward float, increases when pacman eats. Further information about reward func unknown
# info contains lives, dict object

env = gym.make(pacman_hyperparameters["env_name"], 
			   render_mode=pacman_hyperparameters["render_mode"]
			   )

base_path = f"trial_data/trial_{pacman_hyperparameters['trial_number']}"

env = gym.wrappers.RecordVideo(env, f"{base_path}/video/", episode_trigger=lambda t: t % 100 == 99)

actor = Actor(pacman_hyperparameters["device"],
			  pacman_hyperparameters["input_frame_dim"])

critic = Critic(pacman_hyperparameters["device"],
				pacman_hyperparameters["input_frame_dim"])

optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=pacman_hyperparameters["lr"])

obj_func_hist, losses = train(env,
							  actor,
							  critic,
							  optimizer,
							  pacman_hyperparameters["num_episodes"],
							  pacman_hyperparameters["num_actors"],
							  pacman_hyperparameters["num_epochs"],
							  pacman_hyperparameters["eps"],
							  pacman_hyperparameters["gamma"])

json.dump(pacman_hyperparameters, open(f"{base_path}/hyperparameters.json",'w'))


# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)

plot_fancy_loss(obj_func_hist,
				f"{base_path}/rewards.png")

plot_ugly_loss(losses,
			   len(losses),
			   "total_")

torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")

env.close()