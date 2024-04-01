from utils import plot_fancy_loss, load_pickle
from pathlib import Path

# LunarLander-v2
# ALE/MsPacman-v5

# TODO: customize plot function for pacman plots

base_path = f"trials/LunarLander-v2/ppo_q/trial_data/trial_3"

plots = [("rewards", "Reward Projection", "Reward"),
         ("total_loss", "Total loss Projection", "Total loss"),
         ("critic_loss", "Critic loss Projection", "Critic loss"),
         ("ppo_loss", "PPO loss Projection", "PPO loss")]

# check if directory exist, if not, make it
Path(f"{base_path}/plots/").mkdir(parents=True, exist_ok=True)

for name, title, y_label in plots:
	df = load_pickle(f"{base_path}/pickles/{name}{'.pkl'}")
	plot_fancy_loss(df, f"{base_path}/plots/{name}.png", title, y_label)

print("Plots saved successfully!")
