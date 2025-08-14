import json
from pathlib import Path

import torch
import torch.optim as optim

from base import train
from env_factory import EnvFactory
from losses.loss_factory import LossFactory
from models.model_factory import ModelFactory
from utils import dump_to_pickle

# TODO: remove
torch.autograd.set_detect_anomaly(True)

hyperparameters = json.load(open("config.json"))

env_name = hyperparameters["env_name"]

actor, critic = ModelFactory.create_model(
    env_name, hyperparameters["device"], hyperparameters["mode"]
)

actor.to(actor.device)
critic.to(critic.device)

loss_calculator = LossFactory.create_loss_calculator(
    hyperparameters["loss"], hyperparameters["gamma"]
)

base_path = f"trials/{env_name}/{hyperparameters['loss']}/trial_data/trial_{hyperparameters['trial_number']}"

env = EnvFactory.create_env(hyperparameters, base_path, train=True)

optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=hyperparameters["lr"]
)

obj_func_hist, losses, ppo_loss, critic_loss = train(
    env,
    actor,
    critic,
    optimizer,
    hyperparameters["num_episodes"],
    hyperparameters["num_actors"],
    hyperparameters["num_epochs"],
    hyperparameters["num_replay_episodes"],
    hyperparameters["eps"],
    loss_calculator,
    hyperparameters["batch_size"],
)

# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)
Path(f"{base_path}/pickles/").mkdir(parents=True, exist_ok=True)

json.dump(hyperparameters, open(f"{base_path}/hyperparameters.json", "w"))

plots = [
    (obj_func_hist, "/rewards", "Reward Projection", "Reward"),
    (losses, "/total_loss", "Total loss Projection", "Total loss"),
    (critic_loss, "/critic_loss", "Critic loss Projection", "Critic loss"),
    (ppo_loss, "/ppo_loss", "PPO loss Projection", "PPO loss"),
]

for data, name, title, y_label in plots:
    dump_to_pickle(data, f"{base_path}/pickles/{name}.pkl")


torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")

env.close()
