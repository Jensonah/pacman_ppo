import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class __Base_model__(nn.Module):
    def __init__(self, device, mode, lr):
        super(__Base_model__, self).__init__()

        mode_dict = {"state_only": 1, "state_action": 4}

        self.fc1 = nn.Linear(8, 64)

        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, mode_dict[mode])

        self.device = device

        self.optim = torch.optim.Adam(self.parameters(), lr)

    def final_activation(self, x):
        raise NotImplementedError("Final activation not implemented in base class")

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.final_activation(x)

        return x


class Actor(__Base_model__):
    def __init__(self, device, lr):
        super().__init__(device, "state_action", lr)

    # TODO: attach this to environment obj
    def normalize_state(self, state):
        shape = state.shape
        if len(shape) == 2:
            state[:, 0] /= 1.5
            state[:, 1] /= 1.5
            state[:, 2] /= 5.0
            state[:, 3] /= 5.0
            state[:, 4] /= 3.1415927
            state[:, 5] /= 5.0
            # idx 6 and 7 are bools
        else:
            state[0] /= 1.5
            state[1] /= 1.5
            state[2] /= 5.0
            state[3] /= 5.0
            state[4] /= 3.1415927
            state[5] /= 5.0
            # idx 6 and 7 are bools

        return state

    def update(self, loss: Tensor) -> float:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.detach())

    def final_activation(self, x):
        return x

    def act(self, state):
        logits = self.forward(state)

        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)

    def follow_policy(self, x):
        logits = self.forward(x)
        action = torch.argmax(logits, dim=1).detach()
        return int(action), None


class Critic(__Base_model__):
    def __init__(self, device, lr):
        super().__init__(device, "state_only", lr)

    def final_activation(self, x):
        return x

    def update(
        self, loss: Tensor
    ):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.detach())
