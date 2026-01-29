import numpy as np
import random
from typing import Iterator, List

def random_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Randomly samples a batch of clients."""
    for _ in range(total_round):
        yield random.sample(range(n_clients), bs)

def sequence_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters all clients into random groups of batch_size."""
    client_ids = list(range(n_clients))
    random.shuffle(client_ids)
    groups = [client_ids[i:i + bs] for i in range(0, n_clients, bs)]
    # random.shuffle(groups)
    i = -1
    for _ in range(total_round):
        i = (i + 1) % len(groups)
        yield groups[i]

def random_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters all clients into random groups of batch_size."""
    client_ids = list(range(n_clients))
    random.shuffle(client_ids)
    groups = [client_ids[i:i + bs] for i in range(0, n_clients, bs)]
    # random.shuffle(groups)
    i = -1
    for _ in range(total_round):
        # i = (i + 1) % len(groups)
        i = random.randint(0, len(groups)-1)
        yield groups[i]

CS_MAP = {
    'rand': random_sampler,
    'seqflow': sequence_cluster_sampler,
    'edgeflow': random_cluster_sampler
}

def str2ClientSampler(name='rand'):
    return CS_MAP[name]