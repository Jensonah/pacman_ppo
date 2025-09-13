import json
from pathlib import Path

import pandas as pd

from data_models import TrainConfig
from utils import plot_fancy_loss

# LunarLander-v2
# ALE/MsPacman-v5

# TODO: customize plot function for pacman plots

hyperparameters_json = json.load(open("config.json"))

train_config = TrainConfig(**hyperparameters_json)

base_path = f"trials/LunarLander-v3/trial_data/trial_{train_config.trial_number}"

plots = [
    ("obj_val", "Reward Projection", "Reward"),
    ("critic_loss", "Critic loss Projection", "Critic loss"),
    ("ppo_loss", "PPO loss Projection", "PPO loss"),
]

# check if directory exist, if not, make it
Path(f"{base_path}/plots/").mkdir(parents=True, exist_ok=True)

for name, title, y_label in plots:
    df = pd.read_csv(f"{base_path}/data/{name}{'.csv'}")
    plot_fancy_loss(df, f"{base_path}/plots/{name}.png", title, y_label)

print("Plots saved successfully!")
