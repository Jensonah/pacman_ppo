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


class Actor(nn.Module):

	def __init__(self, device):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(8, 120)
		self.fc2 = nn.Linear(120, 240)
		self.fc3 = nn.Linear(240, 120)
		self.fc4 = nn.Linear(120, 4)

		self.device = device
		

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
		# TODO: log_prob or normal prob? Paper says normal prob...
		return action.item(), probs.log_prob(action).exp()
	

class Critic(nn.Module):

	def __init__(self, device):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(8, 120)
		self.fc2 = nn.Linear(120, 240)
		self.fc3 = nn.Linear(240, 120)
		self.fc4 = nn.Linear(120, 1)

		self.device = device
		

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


def collect_episode(env, model):

	episode = []
	terminated = False
	truncated = False

	new_state, info = env.reset()
	
	while not (terminated or truncated):

		state = normalize_state(new_state)
		state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
		
		action, log_probs = model.act(state)

		new_state, reward, terminated, truncated, info = env.step(action)

		episode.append((state, action, log_probs, reward))

	return episode


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


def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, eps, gamma):

	# One style of policy gradient implementation, popularized in [Mni+16] and well-suited for use
	# with recurrent neural networks, runs the policy for T timesteps (where T is much less than the
	# episode length), and uses the collected samples for an update. This style requires an advantage
	# estimator that does not look beyond timestep T .
	# Don't we already have this? Think about this. Or isn't that the entire idea of the estimator?

	obj_func_hist = []
	losses = []

	for _ in tqdm(range(num_episodes)):

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

		for k in range(num_epochs):

			actor_gradient = torch.empty(0)
			critic_gradient = torch.empty(0)

			for j in range(num_actors):

				critic_values_t  = torch.cat([critic(state) for state in states[j]])
				critic_values_t1 = torch.cat((critic_values_t.clone()[1:], torch.zeros(1,1)))

				advantage = gamma*critic_values_t1 + rewards[j] - critic_values_t 
				#advantage *= -1 # flipping because torch minimizes

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


lunar_lander_hyperparameters = {
	"num_episodes" : 300,
	"gamma" : 0.99,
	"lr" : 1e-3,
	"env_name" : "LunarLander-v2",
	"render_mode" : "rgb_array",
	"trial_number" : 4,
	"eps" : 0.2,
	"num_epochs" : 10,
	"num_actors" : 5,
	"device" : "cpu"
}

env = gym.make(lunar_lander_hyperparameters["env_name"], 
			   render_mode=lunar_lander_hyperparameters["render_mode"],
			   continuous=False)

base_path = f"trial_data/trial_{lunar_lander_hyperparameters['trial_number']}"

env = gym.wrappers.RecordVideo(env, f"{base_path}/video/", episode_trigger=lambda t: t % 100 == 99)

actor = Actor(lunar_lander_hyperparameters["device"])

critic = Critic(lunar_lander_hyperparameters["device"])

optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lunar_lander_hyperparameters["lr"])

obj_func_hist, losses = train(env,
							  actor,
							  critic,
							  optimizer,
							  lunar_lander_hyperparameters["num_episodes"],
							  lunar_lander_hyperparameters["num_actors"],
							  lunar_lander_hyperparameters["num_epochs"],
							  lunar_lander_hyperparameters["eps"],
							  lunar_lander_hyperparameters["gamma"])


with open(f'{base_path}/hyperparameters.txt', 'w') as f:
    f.write(str(lunar_lander_hyperparameters))

# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)

plot_fancy_loss(obj_func_hist,
				f"{base_path}/rewards.png")

plot_ugly_loss(losses,
			   len(losses),
			   "total_")

torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")