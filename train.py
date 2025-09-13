import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from base2 import train
from data_models import EpisodeData, TrainConfig
from env_factory import EnvFactory
from models.model_factory import ModelFactory


def get_df(data: list[EpisodeData]) -> pd.DataFrame:
    df = pd.DataFrame(
        {"mean": [d.mean for d in data], "min": [d.min for d in data], "max": [d.max for d in data]}
    )
    return df


hyperparameters_json = json.load(open("config.json"))

train_config = TrainConfig(**hyperparameters_json)

env_name = train_config.env_name

actor, critic = ModelFactory.create_model(env_name, train_config.device, train_config)

actor.to(actor.device)
critic.to(critic.device)


base_path = f"trials/{env_name}//trial_data/trial_{train_config.trial_number}"

env = EnvFactory.create_env(train_config, base_path, train=True)

experiment_data = train(
    env,
    actor,
    critic,
    config=train_config,
)

# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)
Path(f"{base_path}/data/").mkdir(parents=True, exist_ok=True)

json.dump(train_config.model_dump(), open(f"{base_path}/hyperparameters.json", "w"))

obj_val_df = get_df(experiment_data.obj_func_val)
critic_loss_df = get_df(experiment_data.critic_losses)
ppo_loss_df = get_df(experiment_data.ppo_losses)

obj_val_df.to_csv(f"{base_path}/data/obj_val.csv")
critic_loss_df.to_csv(f"{base_path}/data/critic_loss.csv")
ppo_loss_df.to_csv(f"{base_path}/data/ppo_loss.csv")


torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")

env.close()
