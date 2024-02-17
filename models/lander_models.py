import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class __Base_model__(nn.Module):
    
    def __init__(self, device, mode):
        super(__Base_model__, self).__init__()

        mode_dict = {'state_only' : 1, 
                     'state_action' : 4}

        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, mode_dict[mode])

        self.device = device


    def final_activation(self, x):
        raise NotImplementedError("Final activation not implemented in base class")
    

    def normalize_state(self, state):
        state[0] /= 1.5
        state[1] /= 1.5
        state[2] /= 5.0
        state[3] /= 5.0
        state[4] /= 3.1415927
        state[5] /= 5.0	
        # idx 6 and 7 are bools
        
        return state
        

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)
        x = self.final_activation(x)

        return x
    

class Actor(__Base_model__):

    def __init__(self, device, mode):
        super(Actor, self).__init__(device, mode)


    def final_activation(self, x):
        return F.log_softmax(x, dim=1)
    

    def act(self, state):

        probabilities = self.forward(state)

        probs = Categorical(logits=probabilities)
        action = probs.sample()
        return action.item(), probabilities[0, action]
    

class Critic(__Base_model__):

    def __init__(self, device, mode):
        super(Critic, self).__init__(device, mode)
        

    def final_activation(self, x):
        return x