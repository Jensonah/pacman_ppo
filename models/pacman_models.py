import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import cv2


class Base_model(nn.Module):

    def __init__(self, device, mode, input_frame_dim, no_frames, scale):
        super(Base_model, self).__init__()

        if mode != 'state_action':
            raise NotImplementedError("Only state_action mode is implemented")

        # In this implementation the previous frames and the different rgb channels are in the same dimension
        # Reasoning being that on all channels the same kernels will work, and their wouldn't be a need to learn
        # different kernels for each frame

        self.conv1 = nn.Conv2d(3*no_frames, 3*no_frames, (5,5), groups=1, stride=(2,2))
        nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)
        
        self.conv2 = nn.Conv2d(3*no_frames, 2*no_frames, (5,5), groups=1, stride=(1,1))
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2.bias.data.fill_(0.01)
        
        self.conv3 = nn.Conv2d(2*no_frames, no_frames, (5,5), groups=1, stride=(1,1))
        nn.init.kaiming_uniform_(self.conv3.weight)
        self.conv3.bias.data.fill_(0.01)
        
        self.conv4 = nn.Conv2d(no_frames, no_frames, (5,5), groups=1, stride=(1,1))
        nn.init.kaiming_uniform_(self.conv4.weight)
        self.conv4.bias.data.fill_(0.01)
        
        self.conv5 = nn.Conv2d(no_frames, 1, (5,5), groups=1, stride=(1,1))
        nn.init.kaiming_uniform_(self.conv5.weight)
        self.conv5.bias.data.fill_(0.01)
        
        self.flat = nn.Flatten()
        
        self.full1 = nn.Linear(35*22, 500)
        nn.init.kaiming_uniform_(self.full1.weight)
        self.full1.bias.data.fill_(0.01)

        self.full2 = nn.Linear(500, 100)
        nn.init.kaiming_uniform_(self.full2.weight)
        self.full2.bias.data.fill_(0.01)
        
        self.full3 = nn.Linear(100, 9)
        nn.init.xavier_uniform_(self.full3.weight)
        self.full3.bias.data.fill_(0.01)

        self.device = device

        self.input_frame_dim = (input_frame_dim[0], input_frame_dim[1]//scale, input_frame_dim[2]//scale)
        self.no_frames = no_frames

    
    def final_activation(self, x):
        raise NotImplementedError("Final activation not implemented in base class")
        

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
        x = self.final_activation(x)

        return x


class Actor(Base_model):

    def __init__(self, device, mode, input_frame_dim = (3,210,160), no_frames = 2, scale = 2):
        super(Actor, self).__init__(device, mode, input_frame_dim, no_frames, scale)


    def final_activation(self, x):
        return F.log_softmax(x, dim=1)	


    # TODO: these two functions could be placed in a common actor base class

    def act(self, state_3):

        # we bring the last three frames to the model
        # that way the model can see how the ghosts move :)

        probabilities = self.forward(state_3)

        probs = Categorical(logits=probabilities)
        action = probs.sample()
        return action.item(), probabilities[0, action]
    

    def follow_policy(self, x):
        probs = self.forward(x)
        action = torch.argmax(probs).detach()
        return int(action), probs[0, action]
    

    def state_to_normalized_tensor(self, frame):

        dim = self.input_frame_dim

        frame = cv2.resize(frame, (dim[2], dim[1]), interpolation=cv2.INTER_AREA)

        frame = np.array(frame) / 255

        # PyTorch wants the rgb channel first
        transposed_frame = np.transpose(frame, (2, 0, 1))

        return torch.from_numpy(transposed_frame).float().to(self.device)


    def states_generator(self, frames):

        dim = self.input_frame_dim
        zeros_dim = (dim[0]*(self.no_frames-1), dim[1], dim[2])

        zeros = torch.from_numpy(np.zeros(zeros_dim))
        
        state_repr = zeros.float().to(self.device)

        for frame in frames:
            state_repr = torch.cat((state_repr, frame))
            model_ready_state = state_repr.unsqueeze(0)

            yield model_ready_state

            state_repr = state_repr[3:]


    def collect_episode(self, env, on_policy):
        
        terminated = False
        truncated = False

        frames, actions, probs_list, rewards = [], [], [], []

        # (rgb, x, y) -> ((no_frames-1)*rgb, x, y)
        dim = self.input_frame_dim
        zeros_dim = (dim[0]*(self.no_frames-1), dim[1], dim[2])

        zeros = torch.from_numpy(np.zeros(zeros_dim))
        
        state_repr = zeros.float().to(self.device)

        new_state, info = env.reset()

        last_info = info
        
        while not (terminated or truncated):

            frame = self.state_to_normalized_tensor(new_state)
            state_repr = torch.cat((state_repr, frame))
            model_ready_state = state_repr.unsqueeze(0)

            if on_policy:
                action, probs = self.follow_policy(model_ready_state)
            else:
                action, probs = self.act(model_ready_state)

            new_state, reward, terminated, truncated, info = env.step(action)

            if last_info['lives'] > info['lives']:
                reward -= 100

            last_info = info

            # For memory reasons we only save the frame within the state
            # These can be recovered later, using the functions provided
            frames.append(frame), actions.append(action), probs_list.append(probs), rewards.append(reward)

            # 3 because we work with rgb, could also polished by integrating into hyperpars
            # we pop the last frame here
            state_repr = state_repr[3:]

        # Probably unnecessary
        # if truncated:
        #     rewards[-1] -= 50

        return (frames, self.states_generator, actions, probs_list, rewards)
    

class Critic(Base_model):

    def __init__(self, device, mode, input_frame_dim = (3, 210,160), no_frames = 2, scale = 2):
        super(Critic, self).__init__(device, mode, input_frame_dim, no_frames, scale)

    
    def final_activation(self, x):
        return x