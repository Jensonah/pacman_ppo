import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_fancy_loss(df, path, title, y_label):
    means = np.squeeze(df[["mean"]].values)
    mins = np.squeeze(df[["min"]].values)
    maxes = np.squeeze(df[["max"]].values)

    plt.clf()

    xs = range(len(means))

    plt.figure(figsize=(15, 5))

    # Plot means
    plt.plot(xs, means, color="blue", linestyle="solid", alpha=0.8)

    if (
        title == "Reward Projection"
    ):  # "Critic loss Projection" or title == "Total loss Projection":
        line_200 = [200 for _ in range(len(means))]
        line_0 = [0 for _ in range(len(means))]
        plt.plot(xs, line_200, color="red", linestyle="dashed", alpha=0.3)
        plt.plot(xs, line_0, color="red", linestyle="dashed", alpha=0.3)
        ax = plt.gca()
        ax.set_ylim([-750, 350])

    # Plot range
    plt.fill_between(xs, mins, maxes, color="b", alpha=0.3)

    # Set labels
    plt.title(title)
    plt.xlabel("Episode No.")
    plt.ylabel(y_label)
    # plt.legend(['My PPO implementation'])

    plt.savefig(path)


def dump_to_pickle(data, path):
    df = pd.DataFrame(data, columns=["means", "mins", "maxes"])
    df.to_pickle(path)


def flatten(xss):
    return [x for xs in xss for x in xs]


def standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean()
    std = x.std()
    standardized = (x - mean) / std
    return standardized
