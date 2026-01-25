import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import json
from collections import Counter
import seaborn as sns
import scienceplots

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_png(y, filename="curve.png", title="Curve", xlabel="X", ylabel="Y"):
    """
    Saves a PNG image of a curve defined by x and y.

    Args:
        x (list or array): x-axis data
        y (list or array): y-axis data
        filename (str): output filename (should end with .png)
        title (str): plot title
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
    """
    plt.figure(figsize=(6, 4))
    x = np.arange(1, len(y) + 1)
    plt.plot(x, y, label='Curve', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()  # Close the figure to free memory

def save_json(data: dict, filename: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        filename (str): The path to the JSON file (should end with .json).
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def dataset_stats(dict_users, dataset, args = None, n_class=10, n_users = 50):

    # if args is None:
    #     n_class, n_users = 10, 50
    # else:
    #     n_class, n_users = args.n_class, args.n_clients

    # dict users {0: array([], dtype=int64), 1: array([], dtype=int64), ..., 100: array([], dtype=int64)}
    # sns.palplot(sns.color_palette("Accent"))
    sns.set_palette("Set3")
    # plt.style.use('ieee')
    # print(plt.style.available)
    stats = {i: np.array([], dtype='int64') for i in range(len(dict_users))}
    for key, value in dict_users.items():
        for x in value:
            stats[key] = np.concatenate((stats[key], np.array([dataset[x][1]])), axis=0)
    
    nparray = np.zeros([n_class, n_users], dtype = int)
    for j in range(n_users):
        cls = stats[j]
        cls_counter = Counter(cls)
        for i in range(n_class):
            nparray[i][j] = cls_counter[i]

    fig, ax = plt.subplots()
    bottom = np.zeros([n_users], dtype=int)
    for cls in range(n_class):
        ax.bar(range(n_users), nparray[cls], bottom=bottom, label='class{}'.format(cls))
        bottom += nparray[cls]
    # ax.legend(loc='lower right')
    # plt.title('Data Distribution')
    plt.xlabel('Clients')
    plt.ylabel('Amount of Training Data')
    plt.ylim(0,520)
    plt.tight_layout()
    # plt.show()
    plt.savefig('distribution.pdf')