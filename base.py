import torch
import numpy as np
from tqdm import tqdm
from heapq import *
import random
from utils import flatten


def pack_data(data):
    data = np.array(data)
    return (data.mean(), data.min(), data.max())


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
    #[all_probs[i][actions[i]].unsqueeze(0) for i in range(len(actions))]
    return out


def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, num_replay_episodes, eps, loss_calculator, batch_size):

    # We wrap our training in an try-except block such that we can do a Keyboard interrupt
    # without losing our progress

    try:

        obj_func_hist = []
        losses = []
        ppo_losses = []
        critic_losses = []

        pbar = tqdm(range(num_episodes))

        for i in pbar:

            unprocessed_episodes = [actor.collect_episode(env, on_policy=False) for _ in range(num_actors)]
            obj_func_hist.append(pack_data([sum(reward for _, _, _, reward, _ in episode) for episode in unprocessed_episodes]))

            unprocessed_episodes = flatten(unprocessed_episodes)
            random.shuffle(unprocessed_episodes)

            for k in range(num_epochs):

                losses_k = []
                ppo_losses_k = []
                critic_losses_k = []

                # with batch size < num all states code breaks in second loop
                # the variable in question seems to be the prob vector

                for batch_start in range(0, len(unprocessed_episodes), batch_size):

                    batch_end = min(batch_start + batch_size, len(unprocessed_episodes))

                    states, actions, original_probs, rewards, states_t1 = zip(*unprocessed_episodes[batch_start:batch_end])
                    # make data torch ready
                    states = torch.cat(states)
                    original_probs = torch.cat(original_probs).detach()
                    rewards = torch.tensor(rewards).to(actor.device).unsqueeze(1)
                    rewards = get_standardized_tensor(rewards) # will this cause problems in the training?
                    states_t1 = torch.cat(states_t1)

                    critic_values = critic(states)
                    critic_values_t1 = critic(states_t1)

                    loss_calculator.update_losses(critic_values, actions, rewards, critic_values_t1, actor.device)
                    critic_loss = loss_calculator.get_critic_loss()
                    advantage = loss_calculator.get_advantage()

                    actor_probs = get_probs(actor, states, actions)

                    original_probs = original_probs.detach()

                    difference_grad = torch.exp(actor_probs - original_probs).unsqueeze(1)
                    # Note that for k = 0 these are all ones, but we keep the calculation such that the
                    # backtracking algorithm can also see this
                    
                    clipped = torch.clamp(difference_grad, 1 - eps, 1 + eps)

                    ppo_gradient = torch.minimum(difference_grad*advantage, clipped*advantage)
                    ppo_gradient *= -1 # we invert our loss since torch minimizes

                    ppo_loss = torch.mean(ppo_gradient)# mean to normalize by length
                
                    loss = ppo_loss + critic_loss

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    losses_k.append(torch.mean(loss.cpu().detach()))
                    ppo_losses_k.append(torch.mean(ppo_loss.cpu().detach()))
                    critic_losses_k.append(torch.mean(critic_loss.cpu().detach()))

            losses.append(pack_data(losses_k))
            ppo_losses.append(pack_data(ppo_losses_k))
            critic_losses.append(pack_data(critic_losses_k))     

            pbar.set_description(f"Avg. Reward {obj_func_hist[i][0]:.1f}")

        return obj_func_hist, losses, ppo_losses, critic_losses    
    
    except KeyboardInterrupt:        
        # Our training is interrupted, let's save our results
        return obj_func_hist, losses, ppo_losses, critic_losses