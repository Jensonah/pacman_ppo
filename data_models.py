from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field, model_validator


class TrainConfig(BaseModel):
    num_sample_epochs: int
    num_ppo_epochs: int
    num_parallel_actors: int = Field(description="How many episodes to generate in each epoch")
    n_replay_epochs: int = Field(
        description="Replay size in n epochs"
    )
    batch_size: int
    eps: float
    gamma: float
    device: str
    lr_actor: float
    lr_critic: float
    env_name: str
    trial_number: str
    frameskip: int
    repeat_action_probability: float
    obs_type: str


@dataclass
class Experience:
    # Everything here should be detached
    state: torch.Tensor
    action: int
    orig_prob: float
    reward: float
    state_t1: torch.Tensor
    terminated: bool


@dataclass
class EpisodeData:
    mean: float
    min: float
    max: float


@dataclass
class ExperimentData:
    obj_func_val: list[EpisodeData]
    ppo_losses: list[EpisodeData]
    critic_losses: list[EpisodeData]
