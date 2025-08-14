import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, device):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 4)

        self.device = device

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
        x = F.softmax(x, dim=1)

        return x

    def act(self, state):
        probabilities = self.forward(state)

        probs = Categorical(probabilities)
        action = probs.sample()
        # TODO: log_prob or normal prob? PPO paper says normal prob...
        return action.item(), probs.log_prob(action).exp()  # probabilities[0, action]


class Critic(nn.Module):
    def __init__(self, device):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 4)

        self.device = device

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

        return x


def normalize_state(state):
    state[0] /= 1.5
    state[1] /= 1.5
    state[2] /= 5.0
    state[3] /= 5.0
    state[4] /= 3.1415927
    state[5] /= 5.0
    # idx 6 and 7 are bools

    return state


def collect_episode(env, model):
    episode = []
    terminated = False
    truncated = False

    new_state, info = env.reset()

    while not (terminated or truncated):
        state = normalize_state(new_state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)

        action, probs = model.act(state)

        new_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, probs, reward))

    return episode


# index probability of action taken at sampling time with current updated model
def get_probs(actor, states, actions):
    all_probs = torch.cat([actor.forward(state) for state in states])
    actions = torch.Tensor(actions).int()
    out = all_probs[torch.arange(all_probs.size(0)), actions]
    # the line above does the line below more efficiently
    # [all_probs[i][actions[i]].unsqueeze(0) for i in range(len(actions))]
    return out


def get_standardized_tensor(xs):
    # eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    eps = np.finfo(np.float32).eps.item()
    return (xs - xs.mean()) / (xs.std() + eps)


def pack_data(data):
    data = np.array(data)
    return (data.mean(), data.min(), data.max())


def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, eps, gamma):
    # We wrap our training in an try-except block such that we can do a Keyboard interrupt
    # without losing our progress

    try:
        obj_func_hist = []
        losses = []
        ppo_losses = []
        critic_losses = []

        pbar = tqdm(range(num_episodes))

        for i in pbar:
            episodes = [collect_episode(env, actor) for _ in range(num_actors)]

            states = [[state for state, _, _, _ in episode] for episode in episodes]
            actions = [[action for _, action, _, _ in episode] for episode in episodes]
            original_probs = [
                torch.cat([prob for _, _, prob, _ in episode]) for episode in episodes
            ]
            rewards = [
                torch.tensor([reward for _, _, _, reward in episode]).unsqueeze(1)
                for episode in episodes
            ]
            rewards_std = [get_standardized_tensor(reward).float() for reward in rewards]

            losses_k = []
            ppo_losses_k = []
            critic_losses_k = []

            for k in range(num_epochs):
                loss = 0
                ppo_loss = 0
                critic_loss = 0

                for j in range(num_actors):
                    len_episode = len(episodes[j])

                    # TODO: see if we can batch the forwards here
                    # Our net calculates for each state, the value of that state paired with every possible action
                    critic_values_t = torch.cat([critic(state) for state in states[j]])

                    # Here we index for each state the value of that state paired with the best action
                    critic_values_best_t = torch.max(critic_values_t, 1).values.unsqueeze(1)
                    critic_values_best_t1 = torch.cat(
                        (critic_values_best_t.clone()[1:], torch.zeros(1, 1))
                    )

                    # Here we index for each state the value of that state paired with the action taken
                    critic_values_action_t = critic_values_t[
                        torch.arange(len_episode), actions[j]
                    ].unsqueeze(1)
                    critic_values_action_t1 = torch.cat(
                        (critic_values_action_t.clone()[1:], torch.zeros(1, 1))
                    )

                    constant = 1.25
                    mse = (
                        constant
                        * (
                            gamma * critic_values_best_t1
                            - (critic_values_action_t - rewards_std[j])
                        )
                        ** 2
                    )
                    mae = constant * torch.abs(
                        gamma * critic_values_best_t1 - (critic_values_action_t - rewards_std[j])
                    )

                    critic_loss += torch.mean(torch.maximum(mse, mae))

                    # Here we are comparing two state value estimation, one made one timestep later than the other
                    # Their difference is zero if in that timestep we took the optimal action
                    advantage = gamma * critic_values_action_t1 - (
                        critic_values_action_t - rewards_std[j]
                    )

                    if k == 0:
                        actor_probs = original_probs[j].clone()
                        original_probs[j] = original_probs[j].detach()
                    else:
                        actor_probs = get_probs(actor, states[j], actions[j])

                    difference_grad = torch.div(actor_probs, original_probs[j]).unsqueeze(1)
                    # Note that for k = 0 these are all ones, but we keep the calculation such that the
                    # backtracking algorithm can also see this

                    clipped = torch.clamp(difference_grad, 1 - eps, 1 + eps)

                    ppo_gradient = torch.minimum(difference_grad * advantage, clipped * advantage)
                    ppo_gradient *= -1  # we invert our loss since torch minimizes

                    ppo_loss += torch.mean(ppo_gradient)  # mean to normalize by length

                loss = ppo_loss + critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses_k.append(loss.detach() / num_actors)
                ppo_losses_k.append(ppo_loss.detach() / num_actors)
                critic_losses_k.append(critic_loss.detach() / num_actors)

            losses.append(pack_data(losses_k))
            ppo_losses.append(pack_data(ppo_losses_k))
            critic_losses.append(pack_data(critic_losses_k))
            obj_func_hist.append(pack_data([sum(episode) for episode in rewards]))

            pbar.set_description(f"Avg. Reward {obj_func_hist[i][0]:.1f}")

        return obj_func_hist, losses, ppo_losses, critic_losses

    except KeyboardInterrupt:
        # Our training is interrupted, let's save our results
        return obj_func_hist, losses, ppo_losses, critic_losses


def plot_fancy_loss(data, path, title, y_label):
    means, mins, maxes = list(map(list, zip(*data)))

    plt.clf()

    xs = range(len(data))

    # Plot means
    plt.plot(xs, means, color="blue", linestyle="solid", alpha=0.8)

    if title == "Reward Projection":
        line_200 = [200 for _ in range(len(means))]
        line_0 = [0 for _ in range(len(means))]
        plt.plot(xs, line_200, color="red", linestyle="dashed", alpha=0.3)
        plt.plot(xs, line_0, color="red", linestyle="dashed", alpha=0.3)

    # Plot range
    plt.fill_between(xs, mins, maxes, color="b", alpha=0.3)

    # Set labels
    plt.title(title)
    plt.xlabel("Episode No.")
    plt.ylabel(y_label)
    # plt.legend(['My PPO implementation'])
    plt.savefig(path)


lunar_lander_hyperparameters = {
    "num_episodes": 1000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_name": "LunarLander-v2",
    "render_mode": "rgb_array",
    "trial_number": 6,
    "eps": 0.2,
    "num_epochs": 5,
    "num_actors": 5,
    "device": "cpu",
}

env = gym.make(
    lunar_lander_hyperparameters["env_name"],
    render_mode=lunar_lander_hyperparameters["render_mode"],
    continuous=False,
)

base_path = f"trial_data/trial_{lunar_lander_hyperparameters['trial_number']}"

num_actors = lunar_lander_hyperparameters["num_actors"]

env = gym.wrappers.RecordVideo(
    env, f"{base_path}/video/", episode_trigger=lambda t: t % (num_actors * 25) == 0
)

actor = Actor(lunar_lander_hyperparameters["device"])

critic = Critic(lunar_lander_hyperparameters["device"])

optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=lunar_lander_hyperparameters["lr"]
)

obj_func_hist, losses, ppo_loss, critic_loss = train(
    env,
    actor,
    critic,
    optimizer,
    lunar_lander_hyperparameters["num_episodes"],
    lunar_lander_hyperparameters["num_actors"],
    lunar_lander_hyperparameters["num_epochs"],
    lunar_lander_hyperparameters["eps"],
    lunar_lander_hyperparameters["gamma"],
)

json.dump(lunar_lander_hyperparameters, open(f"{base_path}/hyperparameters.json", "w"))

# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)


plots = [
    (obj_func_hist, "/rewards.png", "Reward Projection", "Reward"),
    (losses, "/total_loss.png", "Total loss Projection", "Total loss"),
    (critic_loss, "/critic_loss.png", "Critic loss Projection", "Critic loss"),
    (ppo_loss, "/ppo_loss.png", "PPO loss Projection", "PPO loss"),
]

for data, img_name, title, y_label in plots:
    plot_fancy_loss(data, f"{base_path}{img_name}", title, y_label)

torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")
