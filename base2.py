import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from tqdm import tqdm

from data_models import EpisodeData, Experience, ExperimentData, TrainConfig
from models.lander_models import Actor, Critic
from utils import flatten


def get_episode_data(data: list[float]):
    experience = EpisodeData(mean=np.mean(data), min=np.min(data), max=np.max(data))
    return experience


def normalize_episode(episode, all_time_min, all_time_max) -> list[Experience]:
    def get_normalized_arr(arr, all_time_min, all_time_max):
        # the sum of the rewards should be between 0 and 1 such that our critic can nicely
        # predict it. The best possible trajectory should give 1, the worst 0.
        sum_normalized_arr = (arr - all_time_min / len(arr)) / (all_time_max - all_time_min)
        return sum_normalized_arr

    rewards = np.array([exp.reward for exp in episode])
    normalized_rewards = get_normalized_arr(rewards, all_time_min, all_time_max)

    for i in range(len(episode)):
        episode[i].reward = normalized_rewards[i]

    return episode


def collect_episode(actor: Actor, env, on_policy) -> list[Experience]:
    terminated = False
    truncated = False

    new_state, info = env.reset()
    new_state = actor.normalize_state(new_state)
    new_state = torch.from_numpy(new_state).float().unsqueeze(0).to(actor.device)

    episode = []

    while not (terminated or truncated):
        state = new_state

        if on_policy:
            action, prob = actor.follow_policy(state)
        else:
            action, prob = actor.act(state)
            prob = prob.detach()

        new_state, reward, terminated, truncated, info = env.step(action)

        new_state = actor.normalize_state(new_state)
        new_state = torch.from_numpy(new_state).float().unsqueeze(0).to(actor.device)

        exp = Experience(
            state=state,
            action=action,
            orig_prob=prob,
            reward=reward,
            terminated=terminated,
            state_t1=new_state,
        )

        episode.append(exp)

    return episode


def get_next_state(env, state, action, orig_prob) -> Experience:
    new_state, reward, terminated, truncated, info = env.step(action)
    exp = Experience(
        state=state,
        action=action,
        orig_prob=orig_prob.detach(),
        reward=reward,
        terminated=terminated,
        state_t1=new_state,
    )
    return exp


def collect_episodes(actor: Actor, vec_env: VectorEnv, on_policy) -> list[list[Experience]]:
    ended_ep = np.array([False for _ in range(vec_env.num_envs)])

    states_1, infos = vec_env.reset() # Should we work with seeds here?
    states_t1 = actor.normalize_state(states_1) # Check if this does what I think it does
    states_t1 = torch.from_numpy(states_t1).to(actor.device)

    episodes_collection = [[] for _ in range(vec_env.num_envs)]

    while not ended_ep.all():
        states_t = states_t1

        actions, probs = actor.act(states_t)
        probs = probs.detach()

        actions = actions.cpu().numpy()
        states_t1, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        states_t1 = actor.normalize_state(states_t1) # Check if this does what I think it does
        states_t1 = torch.from_numpy(states_t1).to(actor.device)

        for i, episodes in enumerate(episodes_collection):
            if not ended_ep[i]:
                episode = Experience(
                    state=states_t[i],
                    action=actions[i],
                    orig_prob=probs[i],
                    reward=rewards[i],
                    state_t1=states_t1[i],
                    terminated=terminateds[i]
                )
                episodes.append(episode)

        # We do this calculation here because we still want to add the terminated state_t1
        endeds = np.logical_or(terminateds, truncateds)
        ended_ep = np.logical_or(ended_ep, endeds)

    return episodes_collection


# TODO: import generic actor and critic class as type hints
def train(vec_env, actor: Actor, critic: Critic, config: TrainConfig) -> ExperimentData:
    all_time_min = -1000  # np.inf
    all_time_max = 400  # -np.inf

    all_episodes = []

    experiment_data = ExperimentData([], [], [])

    try:
        pbar = tqdm(range(config.num_sample_epochs))
        for _ in pbar:
            start_sample_time = time.perf_counter()
            episodes = collect_episodes(actor=actor, vec_env=vec_env, on_policy=False)
            end_sample_time = time.perf_counter()
            sample_time = end_sample_time - start_sample_time

            rewards = [sum([exp.reward for exp in episode]) for episode in episodes]
            # normalized_episodes = [
            #     normalize_episode(episode, all_time_min, all_time_max) for episode in episodes
            # ]
            all_episodes.extend(episodes)

            experiment_data.obj_func_val.append(get_episode_data(rewards))
            ppo_losses = []
            critic_losses = []

            # TODO: if its a bottleneck, find a way to do less datacopying
            index = -1*config.n_replay_epochs*config.num_parallel_actors
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

                    critic_loss = critic.update(
                        gamma=config.gamma,
                        batch=batch,
                        value_states=value_states,
                        value_states_t1=value_states_t1,
                    )
                    actor_loss = actor.update(
                        gamma=config.gamma,
                        eps=config.eps,
                        batch=batch,
                        all_probs=actor_probs_log,
                        value_states=value_states,
                        value_states_t1=value_states_t1,
                    )

                    ppo_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

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
