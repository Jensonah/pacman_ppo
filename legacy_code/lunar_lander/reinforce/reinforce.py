from collections import deque
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

lunar_lander_hyperparameters = {
    "num_episodes": 1500,
    "gamma": 0.97,
    "lr": 1e-3,
    "env_name": "LunarLander-v2",
    "render_mode": "rgb_array",
    "state_size": 8,
    "action_size": 4,
    "fc1": 120,
    "fc2": 240,
    "fc3": 120,
    "trial_number": 0,
    "device": "cpu",
    "sliding_window_size": 25,
}

num_episodes = lunar_lander_hyperparameters["num_episodes"]
trial_no_str = str(lunar_lander_hyperparameters["trial_number"])
device = lunar_lander_hyperparameters["device"]


class Pilot(nn.Module):
    def __init__(self, in_dim, h1, h2, h3, out_dim):
        super(Pilot, self).__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.softmax(x, dim=1)

        return x

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probabilities = self.forward(state)
        probs = Categorical(probabilities)
        action = probs.sample()
        return action.item(), probs.log_prob(action)


def normalize_state(state):
    state[0] /= 1.5
    state[1] /= 1.5
    state[2] /= 5.0
    state[3] /= 5.0
    state[4] /= 3.1415927
    state[5] /= 5.0
    # idx 6 and 7 are bools

    return state


def get_episode(env, model):
    episode = []

    # the library technically returns a state observation, but in
    # our case the state and the state observation are equal
    state, info = env.reset(seed=42)
    state = normalize_state(state)

    terminated = False
    truncated = False

    # the library should already include penalties for terminated, for truncated there is no need
    while not (terminated or truncated):
        action, log_probs = model.act(state)

        state, reward, terminated, truncated, info = env.step(action)
        state = normalize_state(state)

        episode.append((log_probs, reward))

    return episode


def get_standardized_tensor(xs):
    ## eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    eps = np.finfo(np.float32).eps.item()
    xs = torch.tensor(xs)
    return (xs - xs.mean()) / (xs.std() + eps)


def plot_fancy_loss(window_size, loss, path):
    # A sliding window that keeps track of the average, minimum and maximum loss
    # Given an array of length N, outputs an array of length N - 2*window_size
    # We do not do padding of any sorts to keep the output array the same length as the input array
    # Complexity is O(N*window_size)
    #
    # I feel like the min and max could be found in lower time complexity
    # but given that this function takes up little of the total runtime,
    # I focus on other things
    def get_plot_data(window_size, loss):
        loss = np.array(loss)

        data = []

        sum_window = loss[:window_size].sum()
        min_window = loss[:window_size].min()
        max_window = loss[:window_size].max()

        data.append((sum_window / window_size, min_window, max_window))

        for i in range(window_size + 1, len(loss) - window_size - 1):
            sum_window -= loss[i - window_size]
            sum_window += loss[i]
            min_window = loss[i - window_size : i].min()
            max_window = loss[i - window_size : i].max()

            data.append((sum_window / window_size, min_window, max_window))

        sum_window = loss[len(loss) - window_size : len(loss)].sum()
        min_window = loss[len(loss) - window_size : len(loss)].min()
        max_window = loss[len(loss) - window_size : len(loss)].max()

        data.append((sum_window / window_size, min_window, max_window))

        return data

    # TODO: Make sure that the final image suggest an unfinished curve

    data = get_plot_data(window_size, loss)

    means, mins, maxes = list(map(list, zip(*data)))

    plt.clf()

    xs = range(window_size, len(loss) - window_size)

    # Plot points
    plt.plot(xs, means, "b", alpha=0.8)
    # plt.plot(pfb_x_mean, pfb_y_mean, 'g', alpha=0.8)

    # Plot errors
    plt.fill_between(xs, mins, maxes, color="b", alpha=0.3)
    # plt.fill_between(pfb_x_mean, pfb_y_low, pfb_y_high, color='g', alpha=0.3)

    # Set labels
    plt.title("Loss projection")
    plt.xlabel("Episode No.")
    plt.ylabel(f"Average sliding window (size {window_size})")
    plt.legend(["My reinforce implementation"])  # , 'PPO for Beginners'])
    # Show graph so user can screenshot
    plt.savefig(path)


def train(num_episodes, env, model, optim, gamma):
    obj_func_hist = []

    for _ in tqdm(range(num_episodes), desc="Epochs"):
        episode = get_episode(env, model)

        total_reward = sum(reward for _, reward in episode)
        obj_func_hist.append(total_reward)

        expected_rewards = deque()

        for t in reversed(range(len(episode))):
            _, reward = episode[t]
            recursive_value = expected_rewards[0] if expected_rewards else 0

            expected_rewards.appendleft(reward + gamma * recursive_value)

        expected_rewards = get_standardized_tensor(expected_rewards)

        gradients = []
        for t in range(len(episode)):
            log_probs, _ = episode[t]
            # negative because torch implements gradient descent
            gradients.append(-log_probs * expected_rewards[t])

        policy_loss = torch.cat(gradients).sum()

        optim.zero_grad()
        policy_loss.backward()
        optim.step()

    return obj_func_hist


env = gym.make(
    lunar_lander_hyperparameters["env_name"],
    render_mode=lunar_lander_hyperparameters["render_mode"],
    continuous=False,
)

base_path = f"trial_data/trial_{lunar_lander_hyperparameters['trial_number']}"

env = gym.wrappers.RecordVideo(env, f"{base_path}/video/", episode_trigger=lambda t: t % 100 == 99)

model = Pilot(
    lunar_lander_hyperparameters["state_size"],
    lunar_lander_hyperparameters["fc1"],
    lunar_lander_hyperparameters["fc2"],
    lunar_lander_hyperparameters["fc3"],
    lunar_lander_hyperparameters["action_size"],
)

optimizer = optim.Adam(model.parameters(), lr=lunar_lander_hyperparameters["lr"])

obj_func_hist = train(num_episodes, env, model, optimizer, lunar_lander_hyperparameters["gamma"])


with open(f"{base_path}/hyperparameters.txt", "w") as f:
    f.write(str(lunar_lander_hyperparameters))


plot_fancy_loss(
    lunar_lander_hyperparameters["sliding_window_size"],
    obj_func_hist,
    f"{base_path}/loss_pretty.png",
)


plt.clf()
plt.plot(list(range(num_episodes)), obj_func_hist)

# check if directoriy exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)

plt.savefig(f"{base_path}/loss.png")

torch.save(model.state_dict(), f"{base_path}/save/model_weights.pt")
