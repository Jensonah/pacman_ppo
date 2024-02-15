import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from pathlib import Path
import cv2
import json

from base import Actor
from base import collect_episode

def write_image(path, tensor):

	arr = np.array(tensor.detach().squeeze())

	image = np.transpose(arr, (1, 2, 0))*255

	# Convert the array to uint8 data type (required by OpenCV)
	image = image.astype(np.uint8)

	# Save the image using OpenCV
	cv2.imwrite(path, cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR))



TRIAL_NUMBER = 6

base_path = f"trial_data/trial_{TRIAL_NUMBER}"

pacman_hyperparameters = json.load(open(f"{base_path}/hyperparameters.json"))

dim = pacman_hyperparameters["input_frame_dim"]
scale = pacman_hyperparameters["scale"]
dim = (dim[0], dim[1]//scale, dim[2]//scale)

# input (210, 160, 3)
# reward float, increases when pacman eats. Further information about reward func unknown
# info contains lives, dict object

env = gym.make(pacman_hyperparameters["env_name"], 
			   render_mode="human",
			   obs_type=pacman_hyperparameters["obs_type"],
			   frameskip=pacman_hyperparameters["frameskip"],
			   repeat_action_probability=pacman_hyperparameters["repeat_action_probability"]
			   )

actor = Actor('cpu',
			  dim,
			  pacman_hyperparameters["no_frames"])

checkpoint = torch.load(f"{base_path}/save/actor_weights.pt", map_location=torch.device('cpu'))

actor.load_state_dict(checkpoint)

episode = collect_episode(env, actor)

# check if directory exist, if not, make it
states_path = f"{base_path}/states/"
# Path(states_path).mkdir(parents=True, exist_ok=True)

# for i in range(len(episode)):
# 	if i % 1 == 0:
# 		write_image(f"{states_path}state_{i}.jpg", episode[i][0])

env.close()
