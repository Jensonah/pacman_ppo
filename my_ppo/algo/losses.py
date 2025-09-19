import torch
import torch.nn.functional as F
from torch import Tensor

from my_ppo.algo.data_models import Experience
from my_ppo.utils import standardize

# TODO: share computations between functions


def get_critic_loss(batch: list[Experience], value_states, value_states_t1, gamma, device) -> Tensor:
    rewards = [exp.reward for exp in batch]
    rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
    terminated = [exp.terminated for exp in batch]
    terminated_t = torch.tensor(terminated, dtype=torch.float).to(device).unsqueeze(1)

    value_states_t1 = value_states_t1.detach()

    loss = F.mse_loss(value_states, rewards_t + gamma * value_states_t1 * (1 - terminated_t))

    return loss


def get_actor_loss(
    batch: list[Experience], all_probs, value_states, value_states_t1, eps, gamma, device
) -> Tensor:
    orig_probs = torch.tensor([exp.orig_prob for exp in batch])

    actions = torch.tensor([exp.action for exp in batch]).to(device).unsqueeze(1)
    action_probs = all_probs.gather(1, actions).squeeze(1)

    diff = torch.exp(action_probs - orig_probs)

    clamped_diff = torch.clamp(diff, 1 - eps, 1 + eps)

    rewards = [exp.reward for exp in batch]
    rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)

    terminated = [exp.terminated for exp in batch]
    terminated_t = torch.tensor(terminated, dtype=torch.float).to(device).unsqueeze(1)

    # How should I handle truncated?
    advantage = rewards_t + gamma * value_states_t1 * (1 - terminated_t) - value_states
    advantage_d = advantage.detach()
    # advantage_d = standardize(advantage_d)

    loss = torch.min(diff * advantage_d, clamped_diff * advantage_d)
    loss = -torch.mean(loss)

    return loss
