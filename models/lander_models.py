import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from data_models import Experience


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

    def update(
        self, batch: list[Experience], all_probs, value_states, value_states_t1, eps, gamma
    ) -> float:
        orig_probs = torch.tensor([exp.orig_prob for exp in batch])

        actions = torch.tensor([exp.action for exp in batch]).to(self.device).unsqueeze(1)
        action_probs = all_probs.gather(1, actions).squeeze(1)

        diff = torch.exp(action_probs - orig_probs)

        clamped_diff = torch.clamp(diff, 1 - eps, 1 + eps)

        rewards = [exp.reward for exp in batch]
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)

        terminated = [exp.terminated for exp in batch]
        terminated_t = torch.tensor(terminated, dtype=torch.float).to(self.device).unsqueeze(1)

        # How should I handle truncated?
        advantage = rewards_t + gamma * value_states_t1 * (1 - terminated_t) - value_states
        advantage_d = advantage.detach()

        loss = torch.min(diff * advantage_d, clamped_diff * advantage_d)
        loss = -torch.mean(loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.detach())

    def final_activation(self, x):
        return x

    # TODO: these two functions could be placed in a common actor base class
    # TODO: the update functions should be refactored into an actor/critic base class

    def act(self, state):
        logits = self.forward(state)

        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)

    def follow_policy(self, x):
        logits = self.forward(x)
        action = torch.argmax(logits, dim=1).detach()
        return int(action), None

    def normalize_state(self, state):
        shape = state.shape
        if len(shape) == 2:
            state[:,0] /= 1.5
            state[:,1] /= 1.5
            state[:,2] /= 5.0
            state[:,3] /= 5.0
            state[:,4] /= 3.1415927
            state[:,5] /= 5.0
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


class Critic(__Base_model__):
    def __init__(self, device, lr):
        super().__init__(device, "state_only", lr)

    def final_activation(self, x):
        return x

    def update(self, batch: list[Experience], value_states, value_states_t1, gamma):
        rewards = [exp.reward for exp in batch]
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        terminated = [exp.terminated for exp in batch]
        terminated_t = torch.tensor(terminated, dtype=torch.float).to(self.device).unsqueeze(1)

        # I could detach states_t1
        value_states_t1 = value_states_t1.detach()

        loss = F.mse_loss(
            value_states,
            rewards_t + gamma * value_states_t1 * (1 - terminated_t)
        )

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.detach())
