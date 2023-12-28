import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):

	def __init__(self, device, input_frame_dim):
		super(Actor, self).__init__()

		# (250,160,3) or 210?
		self.conv1 = nn.Conv2d(9,  18, (3,3), groups=1, stride=(2,1))
		# (124, 158, 6) of 121 156
		self.conv2 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (61, 78, 6) of 63 74
		self.conv3 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (30, 38, 6) why this is 30 and not 29 idk
		self.conv4 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (14, 18, 6)
		self.conv5 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (6, 8, 6)
		self.flat = nn.Flatten()
		# 6*8*6 = 420
		self.full = nn.Linear(5*8*18, 5)
		# 5

		self.device = device
		self.input_frame_dim = input_frame_dim
		

	def forward(self, x):

		x = self.conv1(x)
		x = F.relu(x)

		x = self.conv2(x)
		x = F.relu(x)

		x = self.conv3(x)
		x = F.relu(x)

		x = self.conv4(x)
		x = F.relu(x)

		x = self.conv5(x)
		x = F.relu(x)
		
		x = self.flat(x)

		x = self.full(x)
		x = F.softmax(x, dim=1)

		return x
	

	def act(self, state_3):

		# we bring the last three frames to the model
		# that way the model can see how the ghosts move :)
		# Only the current frame should be attached

		probabilities = self.forward(state_3)
		probs = Categorical(probabilities)
		action = probs.sample()
		# PPO paper says to use normal prob here
		return action.item(), probs.log_prob(action).exp()
	

class Critic(nn.Module):

	def __init__(self, device, input_frame_dim):
		super(Critic, self).__init__()		

		# (250,160,3)
		self.conv1 = nn.Conv2d(9,  18, (3,3), groups=1, stride=(2,1))
		# (124, 158, 6) of 121 156
		self.conv2 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (61, 78, 6) of 63 74
		self.conv3 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (30, 38, 6) why this is 30 and not 29 idk
		self.conv4 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (14, 18, 6)
		self.conv5 = nn.Conv2d(18, 18, (3,3), groups=1, stride=(2,2))
		# (6, 8, 6)
		self.flat = nn.Flatten()
		# 6*8*6 = 420
		self.full = nn.Linear(5*8*18, 1)
		# 1

		self.device = device
		self.input_frame_dim = input_frame_dim
		

	def forward(self, x):

		x = self.conv1(x)
		x = F.relu(x)

		x = self.conv2(x)
		x = F.relu(x)

		x = self.conv3(x)
		x = F.relu(x)

		x = self.conv4(x)
		x = F.relu(x)

		x = self.conv5(x)
		x = F.relu(x)
		
		x = self.flat(x)

		x = self.full(x)
		x = F.sigmoid(x)

		return x


def state_to_normalized_tensor(state, device):

	state = np.array(state) / 255

	# PyTorch wants the rgb channel first
	transposed_array = np.transpose(state, (2, 0, 1))

	return torch.from_numpy(transposed_array).float().to(device)


def collect_episode(env, model):

	episode = []
	terminated = False
	truncated = False

	state2 = torch.from_numpy(np.zeros(model.input_frame_dim)).float().to(model.device)
	state1 = torch.from_numpy(np.zeros(model.input_frame_dim)).float().to(model.device)
	new_state, info = env.reset()

	last_life_value = info['lives']
	
	while not (terminated or truncated):

		state0 = state_to_normalized_tensor(new_state, model.device)

		# state representation
		state_repr = torch.cat((state0, state1, state2)).unsqueeze(0).to(model.device)
		# (9,210,160)

		action, probs = model.act(state_repr)

		new_state, reward, terminated, truncated, info = env.step(action)

		state2 = state1
		state1 = state0

		if last_life_value > info['lives']:
			last_life_value = info['lives']
			# in a previous implementation I always summed the number of current lives to the reward
			# this resulted in pacman hiding in a corner, as staying alive longer -> more rewards
			# now we just give a penalty when he dies
			reward -= 100

		episode.append((state_repr, action, probs, reward))

	return episode
