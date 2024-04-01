import json
from models.model_factory import ModelFactory
import torch
from env_factory import EnvFactory
from tqdm import tqdm

base_path = f"trials/LunarLander-v2/ppo_q/trial_data/trial_0_replay_for_compare"

hyperparameters = json.load(open(f"{base_path}/hyperparameters.json"))

hyperparameters["num_actors"] = 1

actor, _ = ModelFactory.create_model(hyperparameters['env_name'], hyperparameters['device'], hyperparameters['mode'])

actor.load_state_dict(torch.load(f"{base_path}/save/actor_weights.pt", map_location=hyperparameters['device']))

env = EnvFactory.create_env(hyperparameters, base_path, train=False)

episodes = []

# To change render mode see envFactory class
for _ in tqdm(range(250)):
	episodes.append(actor.collect_episode(env, on_policy=True))
	rewards = episodes[-1][-1]
	#print(sum(rewards))


env.close()

rewards_stats = [sum(rewards) for _, _, _, _, rewards in episodes]

print(f"Average reward: {sum(rewards_stats)/len(rewards_stats)}")
print(f"Min reward: {min(rewards_stats)}")
print(f"Max reward: {max(rewards_stats)}")

test_out = {
	"Min reward": min(rewards_stats),
	"Max reward": max(rewards_stats),
	"Average reward": sum(rewards_stats)/len(rewards_stats)
}


json.dump(test_out, open(f"{base_path}/test_out.json",'w'))
