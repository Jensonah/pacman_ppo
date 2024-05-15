import torch
import numpy as np
from tqdm import tqdm
from heapq import *
import random


def pack_data(data):
    data = np.array(data)
    return (data.mean(), data.min(), data.max())


def get_standardized_tensor(xs):
    # eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities   
    eps = np.finfo(np.float32).eps.item()
    return (xs - xs.mean()) / (xs.std() + eps)


def get_probs(actor, states, actions):
    all_probs = torch.cat([actor.forward(state) for state in states])
    actions = torch.Tensor(actions).int().to(actor.device)
    out = all_probs[torch.arange(all_probs.size(0)), actions].to(actor.device)
    # the line above does the line below more efficiently
    #[all_probs[i][actions[i]].unsqueeze(0) for i in range(len(actions))]
    return out


def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, num_replay_episodes, eps, loss_calculator):

    # We wrap our training in an try-except block such that we can do a Keyboard interrupt
    # without losing our progress

    state_generator = actor.state_generator

    try:

        obj_func_hist = []
        losses = []
        ppo_losses = []
        critic_losses = []

        pbar = tqdm(range(num_episodes))

        replay_episodes = []

        for i in pbar:

            non_processed_episodes = [actor.collect_episode(env, on_policy=False) for _ in range(num_actors)]

            episodes = []

            for _a, _b, _c, original_probs, rewards in non_processed_episodes:
                original_probs = torch.cat(original_probs)

                rewards = torch.tensor(rewards).to(actor.device).unsqueeze(1)
                rewards_std = get_standardized_tensor(rewards)

                episodes.append((_a,_b,_c,original_probs, rewards_std))

            losses_k = []
            ppo_losses_k = []
            critic_losses_k = []

            # We will update our models with mini-batches instead of all episodes at the same time
            # We expect that this will improve both computation and convergence time
            # This comes at the cost that now for each state we estimate its value twice instead of once, 
            # but this could also contribute to the aforementioned advantages 
            # We will also have to reimplement replay

            for k in range(num_epochs):

                loss = 0
                ppo_loss = 0
                critic_loss = 0

                # Here we should take random timesteps from random actors
                for j in range(num_actors):

                    compressed_states, states_generator, actions, original_probs, rewards = episodes[j]
                    
                    # TODO: see if we can batch the forwards here
                    #critic_values  = torch.cat([critic(state) for state in states_generator(compressed_states)])
                    critic_values  = critic(torch.cat([state for state in states_generator(compressed_states)]))
                      
                    loss_calculator.update_losses(critic_values, actions, rewards, actor.device)
                    critic_loss += loss_calculator.get_critic_loss()
                    advantage = loss_calculator.get_advantage()

                    if k == 0:
                        actor_probs = original_probs.clone()
                    else:
                        actor_probs = get_probs(actor, states_generator(compressed_states), actions)

                    original_probs = original_probs.detach()

                    difference_grad = torch.exp(actor_probs - original_probs).unsqueeze(1)
                    # Note that for k = 0 these are all ones, but we keep the calculation such that the
                    # backtracking algorithm can also see this
                    
                    clipped = torch.clamp(difference_grad, 1 - eps, 1 + eps)

                    ppo_gradient = torch.minimum(difference_grad*advantage, clipped*advantage)
                    ppo_gradient *= -1 # we invert our loss since torch minimizes

                    ppo_loss += torch.mean(ppo_gradient)# mean to normalize by length
                
                loss = ppo_loss + critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses_k.append(loss.cpu().detach()/num_actors)
                ppo_losses_k.append(ppo_loss.cpu().detach()/num_actors)
                critic_losses_k.append(critic_loss.cpu().detach()/num_actors)

                # new loop
                # idxs = [(i,j) for i in range(num_actors) for j in range(len(num_episodes[i]))]
                # random.shuffle(idxs)

                # for batch_start in range(0, len(idxs), batch_size):

                #     null_entries = []
                #     states = []
                #     states_t1 = []

                #     batch_end = min(batch_start + batch_size, len(idxs)-1)
                #     for k in range(batch_start, batch_end):
                        
                #         i, j = idxs[k]

                #         compressed_state, _, action, original_probs, reward = episodes[i][j]
                #         state = actor.state_generator(state)
                #         states.append(state)

                #         if j != len(episodes[i]) - 1:
                #             compressed_state_t1, _, _, _, _ = episodes[i][j+1]
                #             state_t1 = actor.state_generator(state)  
                #         else:
                #             state_t1 = torch.zeros(4)
                #             null_entries.append((i,j))

                #         states_t1.append(states_t1)
                    
                #     states = torch.cat(states)
                #     states_t1 = torch.cat(states_t1)

                #     critic_values = critic(states)
                #     critic_values_t1 = critic(state_t1)


                
            
            losses.append(pack_data(losses_k))
            ppo_losses.append(pack_data(ppo_losses_k))
            critic_losses.append(pack_data(critic_losses_k))
            obj_func_hist.append(pack_data([sum(rewards) for _, _, _, _, rewards in non_processed_episodes]))

            pbar.set_description(f"Avg. Reward {obj_func_hist[i][0]:.1f}")

        return obj_func_hist, losses, ppo_losses, critic_losses
    
    except KeyboardInterrupt:        
        # Our training is interrupted, let's save our results
        return obj_func_hist, losses, ppo_losses, critic_losses