import json
from utils import plot_fancy_loss, load_pickle


# LunarLander-v2
# ALE/MsPacman-v5

# TODO: modify plot function such that plot is wider
# TODO: customize plot function for pacman plots

base_path = f"trials/LunarLander-v2/ppo_q/trial_data/trial_1"

plots = [("rewards", "Reward Projection", "Reward"),
         ("total_loss", "Total loss Projection", "Total loss"),
         ("critic_loss", "Critic loss Projection", "Critic loss"),
         ("ppo_loss", "PPO loss Projection", "PPO loss")]

for name, title, y_label in plots:
	df = load_pickle(f"{base_path}/{name}{'.pkl'}")
	plot_fancy_loss(df, f"{base_path}/{name}.png", title, y_label)

print("Graphs saved successfully!")
