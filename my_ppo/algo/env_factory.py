import gymnasium as gym

from my_ppo.algo.data_models import TrainConfig


class EnvFactory:

    @classmethod
    def create_env(cls, config: TrainConfig, base_path, train):
        env_name = config.env_name

        if env_name == "LunarLander-v3":
            if train:
                env = gym.make_vec(
                    env_name, num_envs=config.num_parallel_actors, vectorization_mode="sync"
                )
            else:
                env = gym.make(
                    env_name,
                    render_mode="rgb_array",
                )

        elif env_name == "ALE/MsPacman-v5":
            # env = gym.make(
            #     env_name,
            #     render_mode=render_mode,
            #     obs_type=config.obs_type,
            #     frameskip=config.frameskip,
            #     repeat_action_probability=config.repeat_action_probability,
            # )
            raise NotImplementedError("Pacman yet to come")

        else:
            raise NotImplementedError("This environment is not implemented")

        num_actors = config.num_parallel_actors

        if not train:
            env = gym.wrappers.RecordVideo(
                env, f"{base_path}/video", episode_trigger=lambda t: t % (num_actors) == 0
            )

        return env
