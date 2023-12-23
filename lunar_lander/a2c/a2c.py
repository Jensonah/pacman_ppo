import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


lunar_lander_hyperparameters = {
	"num_episodes" : 1500,
	"beta" : 0.9,
	"gamma" : 0.99,
	"lr_actor" : 1e-3,
	"lr_critic" : 1e-3,
	"env_name" : "LunarLander-v2",
	"render_mode" : "rgb_array",
	"state_size" : 8,
	"action_size" : 4,
	"fc1" : 120,
	"fc2" : 240,
	"fc3" : 120,
	"trial_number" : 34,
	"device" : "cpu",
	"sliding_window_size" : 25
}

num_episodes = lunar_lander_hyperparameters["num_episodes"]
trial_no_str = str(lunar_lander_hyperparameters["trial_number"])
device = lunar_lander_hyperparameters["device"]


class Pilot(nn.Module):

	def __init__(self, in_dim, h1, h2, h3, out_dim):
		super(Pilot, self).__init__()
		self.fc1 = nn.Linear(in_dim, h1)
		self.fc2 = nn.Linear(h1, h2)
		self.fc3 = nn.Linear(h2, h3)
		self.fc4 = nn.Linear(h3, out_dim)
		

	def forward(self, x):

		x = self.fc1(x)
		x = F.relu(x)

		x = self.fc2(x)
		x = F.relu(x)

		x = self.fc3(x)
		x = F.relu(x)

		x = self.fc4(x)
		x = F.softmax(x, dim=1)

		return x
	

	def act(self, state):

		probabilities = self.forward(state)
		probs = Categorical(probabilities)
		action = probs.sample()

		return action.item(), probs.log_prob(action)
	

class CoPilot(nn.Module):

	def __init__(self, in_dim, h1, h2, h3):
		super(CoPilot, self).__init__()
		self.fc1 = nn.Linear(in_dim, h1)
		self.fc2 = nn.Linear(h1, h2)
		self.fc3 = nn.Linear(h2, h3)
		self.fc4 = nn.Linear(h3, 1)
		

	def forward(self, x):

		x = self.fc1(x)
		x = F.relu(x)

		x = self.fc2(x)
		x = F.relu(x)

		x = self.fc3(x)
		x = F.relu(x)

		x = self.fc4(x)
		x = F.sigmoid(x)

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



def get_episode(env, model):

	episode = []
	terminated = False
	truncated = False

	new_state, info = env.reset() # TODO: remove seed
	
	while not (terminated or truncated):

		state = normalize_state(new_state)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		
		action, log_probs = model.act(state)

		new_state, reward, terminated, truncated, info = env.step(action)

		episode.append((state, log_probs, reward))

	return episode



def get_standardized_tensor(xs):

	## eps is the smallest representable float, which is 
	# added to the standard deviation of the returns to avoid numerical instabilities   
	eps = np.finfo(np.float32).eps.item()     
	xs = torch.tensor(xs)
	return (xs - xs.mean()) / (xs.std() + eps)



def update_net(gradients, optim, retain):
	
	policy_loss = torch.cat(gradients).sum()
	optim.zero_grad()
	policy_loss.backward(retain_graph=retain)
	optim.step()

	return policy_loss



def train(num_episodes, env, actor, critic, optim_actor, optim_critic, beta, gamma):

	obj_func_hist = []
	losses = []

	for _ in tqdm(range(num_episodes), desc="Epochs"):

		episode = get_episode(env, actor)

		total_reward = sum(reward for _, _, reward in episode)
		obj_func_hist.append(total_reward)

		gradients_actor = []
		gradients_critic = []

		critic_value = [critic.forward(state) for state, _, _ in episode]
		rewards = get_standardized_tensor([reward for _, _, reward in episode])
		log_probs = [prob for _, prob, _ in episode] # scalars

		for t in range(len(episode)):

			V_t = critic_value[t]
			V_t1 = critic_value[t+1] if t < len(episode) - 1 else 0 # safe indexing
			advantage = gamma*V_t1 + rewards[t] - V_t

			# negative because torch implements gradient descent
			gradients_actor.append(-beta*log_probs[t]*advantage) # we can add rewards[t] in here, minus necessary here?

			gradients_critic.append(advantage**2)

		# Both update functions use "expected_rewards[t] - V_s" in their update pass
		# pytorch creates behind the scenes creates computation graphs (maybe look up a course of the inner workings of pytorch)
		# and zero grad removes these then, which then causes a bug when we want to update the critic
		# we could also have summed to losses together and then updated on the actor?

		gradient = gradients_actor + gradients_critic
		loss = update_net(gradient, optim_actor, retain=True)

		losses.append(loss.detach())

	return obj_func_hist, losses


def plot_fancy_loss(window_size, loss, path):

	# A sliding window that keeps track of the average, minimum and maximum loss
	# Given an array of length N, outputs an array of length N - 2*window_size
	# We do not do padding of any sorts to keep the output array the same length as the input array
	# Complexity is O(N*window_size)
	#
	# I feel like the min and max could be found in lower time complexity 
	# but given that this function takes up little of the total runtime, 
	# I focus on other things
	def get_plot_data(window_size, loss):

		loss = np.array(loss)

		data = []

		sum_window = loss[:window_size].sum()
		min_window = loss[:window_size].min()
		max_window = loss[:window_size].max()

		data.append((sum_window/window_size, min_window, max_window))

		for i in range(window_size + 1, len(loss) - window_size - 1):

			sum_window -= loss[i-window_size]
			sum_window += loss[i]
			min_window = loss[i - window_size:i].min()
			max_window = loss[i - window_size:i].max()

			data.append((sum_window/window_size, min_window, max_window))

		sum_window = loss[len(loss)-window_size:len(loss)].sum()
		min_window = loss[len(loss)-window_size:len(loss)].min()
		max_window = loss[len(loss)-window_size:len(loss)].max()

		data.append((sum_window/window_size, min_window, max_window))

		return data

	# TODO: Make sure that the final image suggest an unfinished curve

	data = get_plot_data(window_size, loss)

	means, mins, maxes = list(map(list, zip(*data)))

	plt.clf()

	xs = range(window_size, len(loss) - window_size)

	# Plot points
	plt.plot(xs, means, 'b', alpha=0.8)
	#plt.plot(pfb_x_mean, pfb_y_mean, 'g', alpha=0.8)

	# Plot errors
	plt.fill_between(xs, mins, maxes, color='b', alpha=0.3)
	#plt.fill_between(pfb_x_mean, pfb_y_low, pfb_y_high, color='g', alpha=0.3)

	# Set labels
	plt.title(f'Loss projection')
	plt.xlabel('Episode No.')
	plt.ylabel(f'Average sliding window (size {window_size})')
	plt.legend(['My A2C implementation'])#, 'PPO for Beginners'])
	# Show graph so user can screenshot
	plt.savefig(path)


def plot_ugly_loss(data, length, name):

	plt.clf()
	plt.plot(list(range(length)), data)
	plt.savefig(f"{base_path}/{name}loss.png")



env = gym.make(lunar_lander_hyperparameters["env_name"], 
			   render_mode=lunar_lander_hyperparameters["render_mode"],
			   continuous=False)

base_path = f"trial_data/trial_{lunar_lander_hyperparameters['trial_number']}"

env = gym.wrappers.RecordVideo(env, f"{base_path}/video/", episode_trigger=lambda t: t % 100 == 99)

actor = Pilot(lunar_lander_hyperparameters["state_size"],
			  lunar_lander_hyperparameters["fc1"],
			  lunar_lander_hyperparameters["fc2"],
			  lunar_lander_hyperparameters["fc3"],
			  lunar_lander_hyperparameters["action_size"])

optimizer_actor = optim.Adam(actor.parameters(), lr=lunar_lander_hyperparameters["lr_actor"])

critic = CoPilot(lunar_lander_hyperparameters["state_size"],
			     lunar_lander_hyperparameters["fc1"],
				 lunar_lander_hyperparameters["fc2"],
				 lunar_lander_hyperparameters["fc3"])

optimizer_critic = optim.Adam(critic.parameters(), lr=lunar_lander_hyperparameters["lr_critic"])

obj_func_hist, losses = train(num_episodes, 
								env, 
								actor,
								critic, 
								optimizer_actor,
								optimizer_critic, 
								lunar_lander_hyperparameters["beta"],
								lunar_lander_hyperparameters["gamma"])


with open(f'{base_path}/hyperparameters.txt', 'w') as f:
    f.write(str(lunar_lander_hyperparameters))


# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)


plot_fancy_loss(lunar_lander_hyperparameters["sliding_window_size"], 
				obj_func_hist,
				f"{base_path}/loss_pretty.png")


plot_ugly_loss(losses, num_episodes, "actor_")

torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")