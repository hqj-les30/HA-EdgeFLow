import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


import sampling as sp
from utils import dataset_stats
path_to_data = './data'

def sampling(indices, n_per_class):
    selected = []
    for i in range(len(n_per_class)):
        selected.extend(
            np.random.choice(indices[i], n_per_class[i]).tolist()
        )
    return selected

class Cifar10Local(Dataset):
    data = datasets.CIFAR10(root=path_to_data,
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()]
                            ))

    def __init__(self, indices, n_per_class=None):
        # indices = np.load(path_to_data + '/cifar10_indices.npy', allow_pickle=True)
        # self.indices = sampling(indices, n_per_class)
        if indices is None:
            indices = [i for i in range(len(Cifar10Local.data))]
        self.indices = indices
        self.size = len(self.indices)
        self.targets = [Cifar10Local.data.targets[i] for i in self.indices]

    def __getitem__(self, item):
        return Cifar10Local.data[self.indices[item]][0], Cifar10Local.data[self.indices[item]][1]

    def __len__(self):
        return self.size

class Cifar10Dev(Dataset):
    data = datasets.CIFAR10(root=path_to_data,
                            train=False,
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    def __init__(self):
        self.size = len(Cifar10Dev.data)

    def __getitem__(self, item):
        return Cifar10Dev.data[item][0], Cifar10Dev.data[item][1]

    def __len__(self):
        return self.size

def __create_cifar10_indices():
    from os import path

    if path.exists(path_to_data + '/cifar10_indices.npy'):
        return

    train_data = datasets.CIFAR10(root=path_to_data,
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor())
    indices = [[] for _ in range(10)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)

    np.save(path_to_data + '/cifar10_indices.npy', indices)

class Cifar100Local(Dataset):
    data = datasets.CIFAR100(root=path_to_data,
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()]
                            ))

    def __init__(self, indices, n_per_class=None):
        # indices = np.load(path_to_data + '/cifar100_indices.npy', allow_pickle=True)
        # self.indices = sampling(indices, n_per_class)
        if indices is None:
            indices = [i for i in range(len(Cifar100Local.data))]
        self.indices = indices
        self.size = len(self.indices)
        self.targets = [Cifar100Local.data.targets[i] for i in self.indices]

    def __getitem__(self, item):
        return Cifar100Local.data[self.indices[item]][0], Cifar100Local.data[self.indices[item]][1]

    def __len__(self):
        return self.size

class Cifar100Dev(Dataset):
    data = datasets.CIFAR100(root=path_to_data,
                            train=False,
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    def __init__(self):
        self.size = len(Cifar100Dev.data)

    def __getitem__(self, item):
        return Cifar100Dev.data[item][0], Cifar100Dev.data[item][1]

    def __len__(self):
        return self.size

def __create_cifar100_indices():
    from os import path

    if path.exists(path_to_data + '/cifar100_indices.npy'):
        return

    train_data = datasets.CIFAR100(root=path_to_data,
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor())
    indices = [[] for _ in range(100)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)

    np.save(path_to_data + '/cifar100_indices.npy', indices)

class FashionLocal(Dataset):
    data = datasets.FashionMNIST(root=path_to_data,
                            train=True,
                            download=True,
                            transform=transforms.ToTensor()
                            )

    def __init__(self, indices, n_per_class=None):
        # indices = np.load(path_to_data + '/fashion_indices.npy', allow_pickle=True)
        # self.indices = sampling(indices, n_per_class)
        if indices is None:
            indices = [i for i in range(len(FashionLocal.data))]
        self.indices = indices
        self.size = len(self.indices)
        self.targets = [FashionLocal.data.targets[i] for i in self.indices]

    def __getitem__(self, item):
        return FashionLocal.data[self.indices[item]][0], FashionLocal.data[self.indices[item]][1]

    def __len__(self):
        return self.size

class FashionDev(Dataset):
    data = datasets.FashionMNIST(root=path_to_data,
                            train=False,
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    def __init__(self):
        self.size = len(FashionDev.data)

    def __getitem__(self, item):
        return FashionDev.data[item][0], FashionDev.data[item][1]

    def __len__(self):
        return self.size

def __create_fashion_indices():
    from os import path

    if path.exists(path_to_data + '/fashion_indices.npy'):
        return

    train_data = datasets.FashionMNIST(root=path_to_data,
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor())
    indices = [[] for _ in range(10)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)

    np.save(path_to_data + '/fashion_indices.npy', indices)

def local_data_preparation(
        n_clients = 50,
        dataset = 'cifar10',
        p = [0.4, 0.3, 0.3],
        size_range = (500, 600),
        alpha = 0.8,
        r0 = 0.95,
        r1 = 0.98
):
    n_iid = round(n_clients*p[0])
    n_mix0 = round(n_clients*p[1])
    n_mix1 = n_clients - n_iid - n_mix0

    if dataset == 'cifar10':
        __create_cifar10_indices()
        n_class = 10
        data_func = Cifar10Local
        dev_dataset = Cifar10Dev()
    elif dataset == 'cifar100':
        __create_cifar100_indices()
        n_class = 100
        data_func = Cifar100Local
        dev_dataset = Cifar100Dev()
    elif dataset == 'fashion':
        __create_fashion_indices()
        n_class = 10
        data_func = FashionLocal
        dev_dataset = FashionDev()

    mode = [0] * n_iid + [1] * n_mix0 + [2] * n_mix1

    indices_per_class = np.load(path_to_data + '/' + dataset + '_indices.npy')
    total_data = sum(len(row) for row in indices_per_class)

    size_per_class = indices_per_class.shape[1]
    len_iid, len_mix0, len_mix1 = int(size_per_class * p[0]), int(size_per_class * p[1]), int(size_per_class * p[2])

    iid_clients = {}
    if len_iid > 0:
        index_iid = indices_per_class[:, :len_iid].reshape((n_class * len_iid,))
        ds_iid = data_func(index_iid)
        iid_clients = sp.get_iid(ds_iid, n_iid)
        for id, index in iid_clients.items():
            iid_clients[id] = index_iid[index]
    # dataset_stats(iid_clients, data_func(None), n_users=n_iid)

    mix0_clients = {}
    if len_mix0 > 0:
        index_mix0 = indices_per_class[:, len_iid: len_iid+len_mix0].reshape((n_class * len_mix0,))
        np.random.shuffle(index_mix0)
        ds_mix0 = data_func(index_mix0)
        mix0_clients = sp.get_mixed_noniid(ds_mix0, n_mix0, r0)
        for id, index in mix0_clients.items():
            mix0_clients[id] = index_mix0[index]
    # dataset_stats(mix0_clients, ds_mix0, n_users=n_mix0)

    mix1_clients = {}
    if  len_mix1 > 0:
        index_mix1 = indices_per_class[:, len_iid+len_mix0: len_iid+len_mix0+len_mix1].reshape((n_class * len_mix1,))
        np.random.shuffle(index_mix1)
        ds_mix1 = data_func(index_mix1)
        mix1_clients = sp.get_mixed_noniid(ds_mix1, n_mix1, r1)
        for id, index in mix1_clients.items():
            mix1_clients[id] = index_mix1[index]

    client_datasets = {}
    offset = 0
    for d in [iid_clients, mix0_clients, mix1_clients]:
        for k, v in d.items():
            client_datasets[k + offset] = data_func(v)
            # client_datasets[k+offset] = v
        offset += len(d)

    return client_datasets, mode, dev_dataset, n_class

def proxy_data_preparation(
        size = 500,
        dataset = 'cifar10'
):
    if dataset == 'cifar10':
        data_func = Cifar10Local
    elif dataset == 'cifar100':
        data_func = Cifar100Local
    elif dataset == 'fashion':
        data_func = FashionLocal

    indices = np.load(path_to_data + '/' + dataset + '_indices.npy')
    total_data = sum(len(row) for row in indices)

    idx = np.random.choice(total_data, size)
    return data_func(idx)
if __name__ == '__main__':
    # local_datasets, mode, dev_set, n_class = local_data_preparation(n_clients=100, p=[1.0, 0., 0.], dataset='cifar10', size_range=(500, 600), alpha=0.2)
    # local_datasets, mode, dev_set, n_class = local_data_preparation(n_clients=100, p=[0.1, 0.0, 0.9], dataset='cifar100', size_range=(500, 600), alpha=0.2, r1=1.0)
    local_datasets, mode, dev_set, n_class = local_data_preparation(n_clients=100, p=[0.1, 0.2, 0.7], dataset='cifar100', size_range=(500, 600), alpha=0.2)
    dataset_stats(local_datasets, Cifar100Local(None), n_users=100, n_class=100)



        

        