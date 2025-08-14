import torch


class PPO_Q_loss_calculator:
    def __init__(self, gamma) -> None:
        self.gamma = gamma

        self.critic_values_best_t = None
        self.critic_values_action_t = None

        self.critic_values_best_t1 = None
        self.critic_values_action_t1 = None

        self.rewards = None

        self.retrieved_critic_loss = False
        self.retrieved_advantage = False

    def update_losses(
        self, critic_values, actions, rewards, critic_values_t1, actions_t1, terminated, device
    ):
        len_episode = len(actions)

        self.critic_values_best_t = torch.max(critic_values, 1).values.unsqueeze(1)
        self.critic_values_best_t1 = torch.max(critic_values_t1, 1).values.unsqueeze(1)
        self.critic_values_best_t1 *= terminated

        # Here we index for each state the value of that state paired with the action taken
        self.critic_values_action_t = critic_values[torch.arange(len_episode), actions].unsqueeze(1)
        # This is wrong, action[t+1] does not equal action at t+1
        # self.critic_values_action_t1 = torch.cat((self.critic_values_action_t.clone()[1:], torch.zeros(1,1).to(device)))
        self.critic_values_action_t1 = critic_values[
            torch.arange(len_episode), actions_t1
        ].unsqueeze(1)
        self.critic_values_action_t1 *= terminated

        self.rewards = rewards

        self.retrieved_critic_loss = False
        self.retrieved_advantage = False

    def safety_check(self, already_retrieved):
        if already_retrieved:
            raise RuntimeError("""Losses should be updated using update_losses() before retrieval.
						Make sure to not call this function twice in a row.""")

        return True

    def get_critic_loss(self):
        self.retrieved_critic_loss = self.safety_check(self.retrieved_critic_loss)

        return torch.mean(
            (self.gamma * self.critic_values_best_t1 - (self.critic_values_action_t - self.rewards))
            ** 2
        )

    def get_advantage(self):
        self.retrieved_advantage = self.safety_check(self.retrieved_advantage)

        # Here we are comparing two state value estimation, one made one timestep later than the other
        # Their difference is zero if in that timestep we took the optimal action
        return self.gamma * self.critic_values_action_t1 - (
            self.critic_values_action_t - self.rewards
        )
