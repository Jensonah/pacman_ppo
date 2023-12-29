import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):

	def __init__(self, device, input_frame_dim, no_frames):
		super(Actor, self).__init__()

		# In this implementation the previous frames and the different rgb channels are in the same dimension
		# Reasoning being that on all channels the same kernels will work, and their wouldn't be a need to learn
		# different kernels for each frame

		# (250,160,3) or 210?
		self.conv1 = nn.Conv2d(3*no_frames,  3*2*no_frames, (3,3), groups=1, stride=(2,1))
		# (124, 158, 6) of 121 156
		self.conv2 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (61, 78, 6) of 63 74
		self.conv3 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (30, 38, 6) why this is 30 and not 29 idk
		self.conv4 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (14, 18, 6)
		self.conv5 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (6, 8, 6)
		self.flat = nn.Flatten()
		# 6*8*6 = 420
		self.full = nn.Linear(5*8*3*2*no_frames, 5)
		# 5

		self.device = device
		self.input_frame_dim = input_frame_dim
		self.no_frames = no_frames
		

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

	def __init__(self, device, input_frame_dim, no_frames):
		super(Critic, self).__init__()		

		# (250,160,)
		self.conv1 = nn.Conv2d(3*no_frames,  3*2*no_frames, (3,3), groups=1, stride=(2,1))
		# (124, 158, ...) of 121 156
		self.conv2 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (61, 78, 6) of 63 74
		self.conv3 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (30, 38, 6) why this is 30 and not 29 idk
		self.conv4 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (14, 18, 6)
		self.conv5 = nn.Conv2d(3*2*no_frames, 3*2*no_frames, (3,3), groups=1, stride=(2,2))
		# (6, 8, 6)
		self.flat = nn.Flatten()
		# 6*8*6 = 420
		self.full = nn.Linear(5*8*3*2*no_frames, 1)
		# 1

		self.device = device
		self.input_frame_dim = input_frame_dim
		self.no_frames = no_frames
		

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

	# (rgb, x, y) -> (no_frames*rgb, x, y)
	dim = (model.input_frame_dim[0]*(model.no_frames-1),) + model.input_frame_dim[1:]

	zeros = torch.from_numpy(np.zeros(dim))
	
	state_repr = zeros.float().to(model.device)

	new_state, info = env.reset()

	last_life_value = info['lives']
	
	while not (terminated or truncated):

		new_state = state_to_normalized_tensor(new_state, model.device)
		state_repr = torch.cat((state_repr, new_state))
		model_ready_state = state_repr.unsqueeze(0)

		action, probs = model.act(model_ready_state)

		new_state, reward, terminated, truncated, info = env.step(action)


		if last_life_value > info['lives']:
			last_life_value = info['lives']
			# in a previous implementation I always summed the number of current lives to the reward
			# this resulted in pacman hiding in a corner, as staying alive longer -> more rewards
			# now we just give a penalty when he dies
			reward -= 100

		# in this current implementation we save each frame no_frames times
		# this is a major driver of memory usage
		# this can of course be improved, it would require some more serious refactoring however
		episode.append((model_ready_state, action, probs, reward))

		# 3 because we work with rgb, could also polished by integrating into hyperpars
		# we pop the last frame here
		state_repr = state_repr[3:]

	return episode