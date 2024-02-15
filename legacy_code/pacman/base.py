import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import cv2


class Actor(nn.Module):

	def __init__(self, device, input_frame_dim, no_frames):
		super(Actor, self).__init__()

		# In this implementation the previous frames and the different rgb channels are in the same dimension
		# Reasoning being that on all channels the same kernels will work, and their wouldn't be a need to learn
		# different kernels for each frame

		self.conv1 = nn.Conv2d(3*no_frames, 3*no_frames, (5,5), groups=1, stride=(2,2))
		
		self.conv2 = nn.Conv2d(3*no_frames, 2*no_frames, (5,5), groups=1, stride=(1,1))
		
		self.conv3 = nn.Conv2d(2*no_frames, no_frames, (5,5), groups=1, stride=(1,1))
		
		self.conv4 = nn.Conv2d(no_frames, no_frames, (5,5), groups=1, stride=(1,1))
		
		self.conv5 = nn.Conv2d(no_frames, 1, (5,5), groups=1, stride=(1,1))
		
		self.flat = nn.Flatten()
		
		self.full1 = nn.Linear(35*22, 500)

		self.full2 = nn.Linear(500, 100)
		
		self.full3 = nn.Linear(100, 5)

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

		x = self.full1(x)
		x = F.relu(x)

		x = self.full2(x)
		x = F.relu(x)

		x = self.full3(x)		
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

		# In this implementation the previous frames and the different rgb channels are in the same dimension
		# Reasoning being that on all channels the same kernels will work, and their wouldn't be a need to learn
		# different kernels for each frame

		self.conv1 = nn.Conv2d(3*no_frames, 3*no_frames, (5,5), groups=1, stride=(2,2))
		
		self.conv2 = nn.Conv2d(3*no_frames, 2*no_frames, (5,5), groups=1, stride=(1,1))
		
		self.conv3 = nn.Conv2d(2*no_frames, no_frames, (5,5), groups=1, stride=(1,1))
		
		self.conv4 = nn.Conv2d(no_frames, no_frames, (5,5), groups=1, stride=(1,1))
		
		self.conv5 = nn.Conv2d(no_frames, 1, (5,5), groups=1, stride=(1,1))
		
		self.flat = nn.Flatten()

		self.full1 = nn.Linear(35*22, 500)

		self.full2 = nn.Linear(500, 100)
		
		self.full3 = nn.Linear(100, 1)

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

		x = self.full1(x)
		x = F.relu(x)

		x = self.full2(x)
		x = F.relu(x)

		x = self.full3(x)

		return x


def state_to_normalized_tensor(state, device, dim):

	state = cv2.resize(state, (dim[2], dim[1]), interpolation=cv2.INTER_AREA)

	state = np.array(state) / 255

	# PyTorch wants the rgb channel first
	transposed_array = np.transpose(state, (2, 0, 1))

	return torch.from_numpy(transposed_array).float().to(device)


def get_additional_reward_from_info(last_info, info):
	# if we lost a life we give a penalty
	if last_info['lives'] > info['lives']:
		return -100
	else:
		return 0


def collect_episode(env, model):

	episode = []
	terminated = False
	truncated = False

	# (rgb, x, y) -> ((no_frames-1)*rgb, x, y)
	dim = model.input_frame_dim
	zeros_dim = (dim[0]*(model.no_frames-1), dim[1], dim[2])

	zeros = torch.from_numpy(np.zeros(zeros_dim))
	
	state_repr = zeros.float().to(model.device)

	new_state, info = env.reset()

	last_info = info
	
	while not (terminated or truncated):

		frame = state_to_normalized_tensor(new_state, model.device, dim)
		state_repr = torch.cat((state_repr, frame))
		model_ready_state = state_repr.unsqueeze(0)

		action, probs = model.act(model_ready_state)

		new_state, reward, terminated, truncated, info = env.step(action)

		reward_correction = get_additional_reward_from_info(last_info, info)

		reward += reward_correction

		# For memory reasons we only save the frame within the state
		# These can be recovered later, using the functions provided
		episode.append((frame, action, probs, reward))

		# 3 because we work with rgb, could also polished by integrating into hyperpars
		# we pop the last frame here
		state_repr = state_repr[3:]

	return episode