from my_ppo.algo.data_models import TrainConfig


class ModelFactory:

    @classmethod
    def create_model(cls, env_name, device, config: TrainConfig):
        if env_name == "LunarLander-v3":
            from my_ppo.models.lander_models import Actor, Critic

        elif env_name == "ALE/MsPacman-v5":
            # from models.pacman_models import Actor, Critic
            raise NotImplementedError("Pacman yet to come")

        else:
            raise NotImplementedError("Model not implemented for this environment")

        return Actor(device, config.lr_actor), Critic(device, config.lr_critic)
