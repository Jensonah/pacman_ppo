import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class __Base_model__(nn.Module):
    
    def __init__(self, device, mode):
        super(__Base_model__, self).__init__()

        mode_dict = {'state_only' : 1, 
                     'state_action' : 4}

        self.fc1 = nn.Linear(8, 256)
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(256, 512)
        nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(512, 256)
        nn.init.kaiming_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(256, 128)
        nn.init.kaiming_uniform_(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

        self.fc5 = nn.Linear(128, 128)
        nn.init.kaiming_uniform_(self.fc5.weight)
        self.fc5.bias.data.fill_(0.01)

        self.fc6 = nn.Linear(128, mode_dict[mode])
        nn.init.xavier_uniform_(self.fc6.weight)
        self.fc6.bias.data.fill_(0.01)

        self.device = device


    def final_activation(self, x):
        raise NotImplementedError("Final activation not implemented in base class")
        

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

    def __init__(self, device, _):
        super(Actor, self).__init__(device, "state_action")


    def final_activation(self, x):
        return F.log_softmax(x, dim=1)
    

    # TODO: these two functions could be placed in a common actor base class

    def act(self, state):

        probabilities = self.forward(state)

        probs = Categorical(logits=probabilities)
        action = probs.sample()
        return action.item(), probabilities[0, action]
    

    def follow_policy(self, x):
        probs = self.forward(x)
        action = torch.argmax(probs).detach()
        return int(action), probs[0, action]
    

    def normalize_state(self, state):
        state[0] /= 1.5
        state[1] /= 1.5
        state[2] /= 5.0
        state[3] /= 5.0
        state[4] /= 3.1415927
        state[5] /= 5.0	
        # idx 6 and 7 are bools
        
        return state
    
    def states_generator(self, states):

        for state in states:
            yield state
    

    def collect_episode(self, env, on_policy):

        terminated = False
        truncated = False

        new_state, info = env.reset()

        states, actions, probs_list, rewards = [], [], [], []

        while not (terminated or truncated):

            state = self.normalize_state(new_state)
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            if on_policy:
                action, probs = self.follow_policy(state)
            else:
                action, probs = self.act(state)
                
            new_state, reward, terminated, truncated, info = env.step(action)

            states.append(state), actions.append(action), probs_list.append(probs), rewards.append(reward)

        # Probably unnecessary
        # if truncated:
        #     rewards[-1] -= 50

        return (states, self.states_generator, actions, probs_list, rewards)
    

class Critic(__Base_model__):

    def __init__(self, device, mode):
        super(Critic, self).__init__(device, mode)
        

    def final_activation(self, x):
        return x