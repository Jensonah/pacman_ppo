import json
from utils import plot_fancy_loss, load_pickle

hyperparameters = json.load(open("config.json"))

base_path = f"trials/{hyperparameters['env_name']}/{hyperparameters['loss']}/trial_data/trial_{hyperparameters['trial_number']}"

plots = [("/rewards", "Reward Projection", "Reward"),
         ("/total_loss", "Total loss Projection", "Total loss"),
         ("/critic_loss", "Critic loss Projection", "Critic loss"),
         ("/ppo_loss", "PPO loss Projection", "PPO loss")]

for name, title, y_label in plots:
	df = load_pickle(f"{base_path}/{name}{'.pkl'}")
	plot_fancy_loss(df, f"{base_path}/{name}.png", title, y_label)

