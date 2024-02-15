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


def collect_episode(env, model):

    episode = []
    terminated = False
    truncated = False

    new_state, info = env.reset()

    while not (terminated or truncated):

        state = model.normalize_state(new_state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
        
        action, probs = model.act(state)

        new_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, probs, reward))

    return episode


def train(env, actor, critic, optim, num_episodes, num_actors, num_epochs, eps, gamma, loss_calculator):

    # We wrap our training in an try-except block such that we can do a Keyboard interrupt
    # without losing our progress

    try:

        obj_func_hist = []
        losses = []
        ppo_losses = []
        critic_losses = []

        pbar = tqdm(range(num_episodes))

        for i in pbar:

            episodes = [collect_episode(env, actor) for _ in range(num_actors)]

            states = [[state for state, _, _, _ in episode] for episode in episodes]
            actions = [[action for _, action, _, _ in episode] for episode in episodes]
            original_probs  = [torch.cat([prob for _, _, prob, _ in episode]) for episode in episodes]
            rewards = [torch.tensor([reward for _, _, _, reward in episode]).unsqueeze(1) for episode in episodes]
            rewards_std = [get_standardized_tensor(reward).float() for reward in rewards]

            losses_k = []
            ppo_losses_k = []
            critic_losses_k = []

            for k in range(num_epochs):

                loss = 0
                ppo_loss = 0
                critic_loss = 0

                for j in range(num_actors):

                    len_episode = len(episodes[j])

                    # TODO: see if we can batch the forwards here
                    critic_values  = torch.cat([critic(state) for state in states[j]])
                    
                    loss_calculator.update_losses(critic_values, actions[j], rewards_std[j])
                    critic_loss += loss_calculator.get_critic_loss()
                    advantage = loss_calculator.get_advantage()

                    if k == 0:
                        actor_probs = original_probs[j].clone()
                        original_probs[j] = original_probs[j].detach()
                    else:
                        actor_probs = get_probs(actor, states[j], actions[j])

                    difference_grad = torch.div(actor_probs, original_probs[j]).unsqueeze(1)
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
            obj_func_hist.append(pack_data([sum(episode) for episode in rewards]))

            pbar.set_description(f"Avg. Reward {obj_func_hist[i][0]:.1f}")

        return obj_func_hist, losses, ppo_losses, critic_losses
    
    except KeyboardInterrupt:        
        # Our training is interrupted, let's save our results
        return obj_func_hist, losses, ppo_losses, critic_losses