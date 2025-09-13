import random
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import flatten


def pack_data(data):
    data = np.array(data)
    return (data.mean(), data.min(), data.max())


def get_normalized_tensor(arr, all_time_min, all_time_max):
    # the sum of the rewards should be between 0 and 1 such that our critic can nicely predict it
    # The best possible trajectory should give 1
    sum_normalized_arr = (arr - all_time_min/len(arr)) / (all_time_max - all_time_min)
    return sum_normalized_arr


def get_standardized_tensor(xs):
    # eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    eps = np.finfo(np.float32).eps.item()
    return (xs - xs.mean()) / (xs.std() + eps)


def get_probs(actor, states, actions):
    all_probs = actor.forward(states)
    actions = torch.Tensor(actions).int().to(actor.device)
    out = all_probs[torch.arange(all_probs.size(0)), actions].to(actor.device)
    # the line above does the line below more efficiently
    # [all_probs[i][actions[i]].unsqueeze(0) for i in range(len(actions))]
    return out


def train(
    env,
    actor,
    critic,
    optim,
    num_episodes,
    num_actors,
    num_epochs,
    num_replay_episodes,
    eps,
    loss_calculator,
    batch_size,
):
    # We wrap our training in an try-except block such that we can do a Keyboard interrupt
    # without losing our progress

    try:
        obj_func_hist = []
        losses = []
        ppo_losses = []
        critic_losses = []

        pbar = tqdm(range(num_episodes))

        all_time_min = -1000 # np.inf #-1_000
        all_time_max = 400 #-np.inf # 400

        sample_time = 0
        cpu_time = 0
        gpu_time = 0

        for i in pbar:

            start_sample_time = time.perf_counter()

            # TODO: we can parallelize this
            unprocessed_episodes = [
                actor.collect_episode(env, on_policy=False) for _ in range(num_actors)
            ]

            end_sample_time = time.perf_counter()
            sample_time += end_sample_time - start_sample_time

            obj_func_hist.append(
                pack_data(
                    [
                        sum(reward for _, _, _, reward, _, _, _ in episode)
                        for episode in unprocessed_episodes
                    ]
                )
            )

            all_time_min = min(all_time_min, obj_func_hist[-1][1])
            all_time_max = max(all_time_max, obj_func_hist[-1][2])

            # we were not normalising for each episode separately, but for all episodes together...
            for j in range(len(unprocessed_episodes)):
                episode = unprocessed_episodes[j]
                rewards = [reward for _, _, _, reward, _, _, _ in episode]
                rewards = get_normalized_tensor(
                    torch.tensor(rewards).to(actor.device).unsqueeze(1),
                    all_time_min,
                    all_time_max,
                )
                normalised_episode = [
                    (state, action, prob, rewards[k], state_t1, action_t1, terminated)
                    for k, (state, action, prob, _, state_t1, action_t1, terminated) in enumerate(episode)
                ]
                unprocessed_episodes[j] = normalised_episode


            unprocessed_episodes = flatten(unprocessed_episodes)
            random.shuffle(unprocessed_episodes)

            states, actions, original_probs, rewards, states_t1, actions_t1, terminated = (
                zip(*unprocessed_episodes)
            )
            states = torch.cat(states)
            original_probs = torch.cat(original_probs).detach()
            rewards = torch.tensor(rewards).to(actor.device).unsqueeze(1)
            #rewards = get_normalized_tensor(rewards, all_time_min, all_time_max)
            # rewards = get_standardized_tensor(rewards, all_time_min, all_time_max) # will this cause problems in the training? Is this
            states_t1 = torch.cat(states_t1)
            terminated = torch.tensor(
                (~np.array(terminated, dtype=bool)).astype("float")
            ).unsqueeze(1)

            end_cpu_time = time.perf_counter()
            cpu_time += end_cpu_time - end_sample_time

            # TODO: we can bring more operations to the GPU

            losses_k = []
            ppo_losses_k = []
            critic_losses_k = []
            for _ in range(num_epochs):

                for batch_start in range(0, len(unprocessed_episodes), batch_size):
                    start_cpu_time = time.perf_counter()

                    # Not sure if I should keep this
                    batch_end = min(batch_start + batch_size, len(unprocessed_episodes))
                    if batch_end - batch_start < batch_size:
                        continue

                    states_b = states[batch_start:batch_end]
                    actions_b = actions[batch_start:batch_end]
                    original_probs_b = original_probs[batch_start:batch_end]
                    rewards_b = rewards[batch_start:batch_end]
                    states_t1_b = states_t1[batch_start:batch_end]
                    actions_t1_b = actions_t1[batch_start:batch_end]
                    terminated_b = terminated[batch_start:batch_end]
                    # make data torch ready -> put this outside batch loop?

                    # Doing critic twice is also expensive
                    critic_values = critic(states_b)
                    critic_values_t1 = critic(states_t1_b)

                    loss_calculator.update_losses(
                        critic_values,
                        actions_b,
                        rewards_b,
                        critic_values_t1,
                        actions_t1_b,
                        terminated_b,
                        actor.device,
                    )
                    end_cpu_time = time.perf_counter()
                    cpu_time += end_cpu_time - start_cpu_time

                    critic_loss = loss_calculator.get_critic_loss()
                    advantage = loss_calculator.get_advantage()
                    advantage = get_standardized_tensor(advantage)
                    # It makes sense that if the advantage does not contain any negative values,
                    # all actions will be reinforced as we minimize.
                    # I do not understand why ppo loss stays stable at 0.71 however

                    actor_probs = get_probs(actor, states_b, actions_b)

                    # These are all ones
                    difference_grad = torch.exp(actor_probs - original_probs_b).unsqueeze(0)
                    # Note that for k = 0 these are all ones, but we keep the calculation such that
                    # the backtracking algorithm can also see this

                    clipped = torch.clamp(difference_grad, 1 - eps, 1 + eps)

                    # error here?
                    ppo_gradient = torch.minimum(difference_grad * advantage, clipped * advantage)
                    ppo_gradient *= -1  # we invert our loss since torch minimizes

                    ppo_loss = torch.mean(
                        ppo_gradient
                    )

                    loss = ppo_loss + critic_loss

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    end_gpu_time = time.perf_counter()
                    gpu_time += end_gpu_time - end_cpu_time

                    losses_k.append(torch.mean(loss.cpu().detach()))
                    ppo_losses_k.append(torch.mean(ppo_loss.cpu().detach()))
                    critic_losses_k.append(torch.mean(critic_loss.cpu().detach()))

                    pbar.set_description(
                        f"Avg. Reward {obj_func_hist[i][0]:.1f}, "
                        f"Sampling Time {sample_time:.2f}s, "
                        f"CPU Time {cpu_time:.2f}s, "
                        f"GPU Time {gpu_time:.2f}s"
                    )

            losses.append(pack_data(losses_k))
            ppo_losses.append(pack_data(ppo_losses_k))
            critic_losses.append(pack_data(critic_losses_k))

        return obj_func_hist, losses, ppo_losses, critic_losses

    except KeyboardInterrupt:
        # Our training is interrupted, let's save our results
        return obj_func_hist, losses, ppo_losses, critic_losses
