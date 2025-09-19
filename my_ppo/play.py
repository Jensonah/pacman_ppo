import json

import torch
from tqdm import tqdm

from my_ppo.algo.data_models import TrainConfig
from my_ppo.algo.env_factory import EnvFactory
from my_ppo.algo.episodes import collect_episode
from my_ppo.models.model_factory import ModelFactory

base_path = "trials/LunarLander-v3/trial_data/trial_replay"

hyperparameters_json = json.load(open("config.json"))

train_config = TrainConfig(**hyperparameters_json)
train_config.num_parallel_actors = 1

env_name = train_config.env_name

actor, critic = ModelFactory.create_model(env_name, train_config.device, train_config)

actor.load_state_dict(
    torch.load(f"{base_path}/save/actor_weights.pt", map_location=train_config.device)
)

env = EnvFactory.create_env(train_config, base_path, train=False)

episodes = []

# To change render mode see envFactory class
for _ in tqdm(range(10)):
    episodes.append(collect_episode(actor, env, on_policy=True))
    rewards = episodes[-1][-1]
    # print(sum(rewards))


env.close()

rewards = [[exp.reward for exp in experience] for experience in episodes]

rewards_stats = [sum(r) for r in rewards]

print(f"Average reward: {sum(rewards_stats) / len(rewards_stats)}")
print(f"Min reward: {min(rewards_stats)}")
print(f"Max reward: {max(rewards_stats)}")

test_out = {
    "Min reward": min(rewards_stats),
    "Max reward": max(rewards_stats),
    "Average reward": sum(rewards_stats) / len(rewards_stats),
}


json.dump(test_out, open(f"{base_path}/test_out.json", "w"))
