import torch
import torch.optim as optim
import gymnasium as gym
from pathlib import Path
import json
from base import train
from models.model_factory import ModelFactory
from losses.loss_factory import LossFactory
from utils import plot_fancy_loss, dump_to_pickle


hyperparameters = json.load(open("config.json"))

env_name = hyperparameters["env_name"]

actor, critic = ModelFactory.create_model(env_name, hyperparameters["device"], hyperparameters['mode'])

loss_calculator = LossFactory.create_loss_calculator(hyperparameters["loss"], hyperparameters["gamma"])

env = gym.make(hyperparameters["env_name"], 
               render_mode=hyperparameters["render_mode"],
               continuous=False)

base_path = f"trials/{env_name}/{hyperparameters['loss']}/trial_data/trial_{hyperparameters['trial_number']}"

num_actors = hyperparameters["num_actors"]

env = gym.wrappers.RecordVideo(env, f"{base_path}/video/", episode_trigger=lambda t: t % (num_actors*25) == 0)

optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=hyperparameters["lr"])

obj_func_hist, losses, ppo_loss, critic_loss = train(env,
                              actor,
                              critic,
                              optimizer,
                              hyperparameters["num_episodes"],
                              num_actors,
                              hyperparameters["num_epochs"],
                              hyperparameters["eps"],
                              hyperparameters["gamma"],
							  loss_calculator)

json.dump(hyperparameters, open(f"{base_path}/hyperparameters.json",'w'))

# check if directory exist, if not, make it
Path(f"{base_path}/save/").mkdir(parents=True, exist_ok=True)

plots = [(obj_func_hist, "/rewards", "Reward Projection", "Reward"),
         (losses, "/total_loss", "Total loss Projection", "Total loss"),
         (critic_loss, "/critic_loss", "Critic loss Projection", "Critic loss"),
         (ppo_loss, "/ppo_loss", "PPO loss Projection", "PPO loss")]

for data, name, title, y_label in plots:
    dump_to_pickle(data, f"{base_path}{name}.pkl")   
    

torch.save(actor.state_dict(), f"{base_path}/save/actor_weights.pt")
torch.save(critic.state_dict(), f"{base_path}/save/critic_weights.pt")

env.close()