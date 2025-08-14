import gymnasium as gym


class EnvFactory:
    def create_env(hyperparameters, base_path, train):
        env_name = hyperparameters["env_name"]

        if train == True:
            render_mode = hyperparameters["render_mode"]
            out_path = f"{base_path}/video_train/"
            video_trigger = 25
        else:
            render_mode = "None"  # "Human"
            out_path = f"{base_path}/video_on_policy/"
            video_trigger = 1

        if env_name == "LunarLander-v2":
            env = gym.make(env_name, render_mode=render_mode)

        elif env_name == "ALE/MsPacman-v5":
            env = gym.make(
                env_name,
                render_mode=render_mode,
                obs_type=hyperparameters["obs_type"],
                frameskip=hyperparameters["frameskip"],
                repeat_action_probability=hyperparameters["repeat_action_probability"],
            )

        else:
            raise NotImplementedError("This environment is not implemented")

        num_actors = hyperparameters["num_actors"]

        # env = gym.wrappers.RecordVideo(env, out_path, episode_trigger=lambda t: t % (num_actors*video_trigger) == 0)

        return env
