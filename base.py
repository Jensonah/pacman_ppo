import torch
import numpy as np
from tqdm import tqdm


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
    actions = torch.Tensor(actions).int()
    out = all_probs[torch.arange(all_probs.size(0)), actions]
    # the line above does the line below more efficiently
    #[all_probs[i][actions[i]].unsqueeze(0) for i in range(len(actions))]
    return out


def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, eps, loss_calculator):

    # We wrap our training in an try-except block such that we can do a Keyboard interrupt
    # without losing our progress

    try:

        obj_func_hist = []
        losses = []
        ppo_losses = []
        critic_losses = []

        pbar = tqdm(range(num_episodes))

        for i in pbar:

            episodes = [actor.collect_episode(env) for _ in range(num_actors)]

            losses_k = []
            ppo_losses_k = []
            critic_losses_k = []

            for k in range(num_epochs):

                loss = 0
                ppo_loss = 0
                critic_loss = 0

                for j in range(num_actors):

                    compressed_states, states_generator, actions, original_probs, rewards = episodes[j]

                    original_probs = torch.tensor(original_probs).to(actor.device)
                    rewards = torch.tensor(rewards).to(actor.device) 

                    rewards_std = get_standardized_tensor(rewards)

                    # TODO: see if we can batch the forwards here
                    critic_values  = torch.cat([critic(state) for state in states_generator(compressed_states)])
                    
                    loss_calculator.update_losses(critic_values, actions, rewards_std)
                    critic_loss += loss_calculator.get_critic_loss()
                    advantage = loss_calculator.get_advantage()

                    if k == 0:
                        actor_probs = original_probs.clone()
                        original_probs = original_probs.detach()
                    else:
                        actor_probs = get_probs(actor, states_generator(compressed_states), actions)

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

                losses_k.append(loss.detach()/num_actors)
                ppo_losses_k.append(ppo_loss.detach()/num_actors)
                critic_losses_k.append(critic_loss.detach()/num_actors)
            
            losses.append(pack_data(losses_k))
            ppo_losses.append(pack_data(ppo_losses_k))
            critic_losses.append(pack_data(critic_losses_k))
            obj_func_hist.append(pack_data([sum(rewards) for _, _, _, _, rewards in episodes]))

            pbar.set_description(f"Avg. Reward {obj_func_hist[i][0]:.1f}")

        return obj_func_hist, losses, ppo_losses, critic_losses
    
    except KeyboardInterrupt:        
        # Our training is interrupted, let's save our results
        return obj_func_hist, losses, ppo_losses, critic_losses