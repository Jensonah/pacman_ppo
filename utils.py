import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_fancy_loss(df, path, title, y_label):

    means = np.squeeze(df[['means']].values)
    mins = np.squeeze(df[['mins']].values)
    maxes = np.squeeze(df[['maxes']].values)
    
    plt.clf()

    xs = range(len(means))

    # Plot means
    plt.plot(xs, means,     color='blue', linestyle='solid', alpha=0.8)

    if title == "Reward Projection":
        line_200 =  [200 for _ in range(len(means))]
        line_0   =  [0   for _ in range(len(means))]
        plt.plot(xs, line_200,  color='red',  linestyle='dashed', alpha=0.3)
        plt.plot(xs, line_0,    color='red',  linestyle='dashed', alpha=0.3)


    # Plot range
    plt.fill_between(xs, mins, maxes, color='b', alpha=0.3)

    # Set labels
    plt.title(title)
    plt.xlabel('Episode No.')
    plt.ylabel(y_label)
    #plt.legend(['My PPO implementation'])
    plt.savefig(path)
    

def dump_to_pickle(data, path):
    df = pd.DataFrame(data, columns=['means', 'mins', 'maxes'])
    df.to_pickle(path)


def load_pickle(path):
    df = pd.read_pickle(path)
    return df