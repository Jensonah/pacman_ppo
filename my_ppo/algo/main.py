import random
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from my_ppo.algo.data_models import ExperimentData, TrainConfig
from my_ppo.algo.episodes import collect_episodes, get_episode_data
from my_ppo.algo.losses import get_actor_loss, get_critic_loss
from my_ppo.models.lander_models import Actor, Critic
from my_ppo.utils import flatten


# TODO: play around with episode/advantage normalization/standardization
def train(vec_env, actor: Actor, critic: Critic, config: TrainConfig) -> ExperimentData:
    all_episodes = []

    experiment_data = ExperimentData([], [], [])

    try:
        pbar = tqdm(range(config.num_sample_epochs))
        for _ in pbar:
            start_sample_time = time.perf_counter()
            episodes = collect_episodes(actor=actor, vec_env=vec_env)
            end_sample_time = time.perf_counter()
            sample_time = end_sample_time - start_sample_time

            rewards = [sum([exp.reward for exp in episode]) for episode in episodes]
            all_episodes.extend(episodes)

            experiment_data.obj_func_val.append(get_episode_data(rewards))
            ppo_losses = []
            critic_losses = []

            # TODO: if its a bottleneck, find a way to do less datacopying
            index = -1 * config.n_replay_epochs * config.num_parallel_actors
            epoch_episodes = all_episodes[index:]
            epoch_episodes = flatten(epoch_episodes)
            for _ in range(config.num_ppo_epochs):
                random.shuffle(epoch_episodes)

                for batch_start in range(0, len(epoch_episodes), config.batch_size):
                    if batch_start + config.batch_size > len(epoch_episodes):
                        break
                    batch = epoch_episodes[batch_start : batch_start + config.batch_size]

                    states = torch.stack([exp.state for exp in batch], dim=0)
                    value_states = critic.forward(states)

                    states_t1 = torch.stack([exp.state_t1 for exp in batch], dim=0)
                    value_states_t1 = critic.forward(states_t1)

                    actor_probs = actor.forward(states)
                    actor_probs_log = F.log_softmax(actor_probs, dim=1)

                    critic_loss = get_critic_loss(
                        gamma=config.gamma,
                        batch=batch,
                        value_states=value_states,
                        value_states_t1=value_states_t1,
                        device=critic.device,
                    )
                    actor_loss = get_actor_loss(
                        gamma=config.gamma,
                        eps=config.eps,
                        batch=batch,
                        all_probs=actor_probs_log,
                        value_states=value_states,
                        value_states_t1=value_states_t1,
                        device=actor.device,
                    )

                    critic.update(critic_loss)
                    actor.update(actor_loss)

                    ppo_losses.append(actor_loss.detach())
                    critic_losses.append(critic_loss.detach())

            calc_time = time.perf_counter() - end_sample_time

            pbar.set_description(
                f"Reward: {experiment_data.obj_func_val[-1].mean:.1f}, "
                f"Sample time: {sample_time:.1f}s, Calc time: {calc_time:.1f}s"
            )
            experiment_data.ppo_losses.append(get_episode_data(ppo_losses))
            experiment_data.critic_losses.append(get_episode_data(critic_losses))

        return experiment_data

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress...")
        return experiment_data
