import numpy as np
import torch
from gymnasium.vector import VectorEnv

from my_ppo.algo.data_models import EpisodeData, Experience
from my_ppo.models.lander_models import Actor


def get_episode_data(data: list[float]):
    experience = EpisodeData(mean=np.mean(data), min=np.min(data), max=np.max(data))
    return experience


# Used for final video
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


# Used for training
def collect_episodes(actor: Actor, vec_env: VectorEnv) -> list[list[Experience]]:
    ended_ep = np.array([False for _ in range(vec_env.num_envs)])

    states_1, infos = vec_env.reset()
    states_t1 = actor.normalize_state(states_1)
    states_t1 = torch.from_numpy(states_t1).to(actor.device)

    episodes_collection = [[] for _ in range(vec_env.num_envs)]

    while not ended_ep.all():
        states_t = states_t1

        actions, probs = actor.act(states_t)
        probs = probs.detach()

        actions = actions.cpu().numpy()
        states_t1, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        states_t1 = actor.normalize_state(states_t1)
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
