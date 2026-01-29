import argparse
import os
import numpy as np

from server import Server, Server_Configs, Server_Logger
from client import client_preparation
from client_sampler import str2ClientSampler
from models import str2model_fn
from data_wrapper import local_data_preparation, proxy_data_preparation

from utils import *
def run_classification(args):
    p = [float(x) for x in args.p.strip('[]').split('/')]
    # print(p)
    local_datasets, mode_vec, test_dataset, n_class = local_data_preparation(
        n_clients= args.n_clients,
        dataset= args.dataset,
        p=p,
        size_range= (500, 501),
        r1 = args.r1
    )
    proxy_dataset = proxy_data_preparation(
        size = 2000,
        dataset = args.dataset
    )
    client_vec = client_preparation(
        local_datasets, mode_vec, args
    )
    args.model_fn = str2model_fn(args.dataset)
    args.client_sampler = str2ClientSampler(args.selection)
    args.n_class = n_class
    serverconfigs = Server_Configs(args)
    serverlogger = Server_Logger(test_dataset)
    server = Server(
        client_vec, mode_vec, serverconfigs, serverlogger, args.exp_id, proxy_dataset
    )
    
    server.run()
    loss, acc = serverlogger.average_loss, serverlogger.global_accuracy
    runtime = serverlogger.end_time - serverlogger.start_time
    log = serverlogger.log

    return loss, acc, runtime, log



def main(args):
    os.makedirs(args.path_to_save, exist_ok=True)

    set_seed(args.seed)
    loss, acc, runtime, log = run_classification(args)
    loss, acc = np.array(loss), np.array(acc)
    np.save(
        args.path_to_save + 'loss_{}_{}_{}_R[{}]_l_[{}]_bs_[{}]_le[{}]_acc[{:.4f}]_time[{:.3f}].npy'.format(args.dataset, args.exp_id, args.selection, args.global_epoch, args.l, args.selection_size, args.local_epoch, acc[-1], runtime),
        loss
    )
    save_png(
        loss,
        args.path_to_save + 'loss_{}_{}_{}_R[{}]_l_[{}]_bs_[{}]_le[{}]_acc[{:.4f}]_time[{:.3f}].png'.format(args.dataset, args.exp_id, args.selection, args.global_epoch, args.l, args.selection_size, args.local_epoch, acc[-1], runtime),
        'Client Training Loss',
        'Round',
        "loss"
    )
    np.save(
        args.path_to_save + 'acc_{}_{}_{}_R[{}]_l_[{}]_bs_[{}]_le[{}]_acc[{:.4f}]_time[{:.3f}].npy'.format(args.dataset, args.exp_id, args.selection, args.global_epoch, args.l, args.selection_size, args.local_epoch, acc[-1], runtime),
        acc
    )
    save_png(
        acc,
        args.path_to_save + 'acc_{}_{}_{}_R[{}]_l_[{}]_bs_[{}]_le[{}]_acc[{:.4f}]_time[{:.3f}].png'.format(args.dataset, args.exp_id, args.selection, args.global_epoch, args.l, args.selection_size, args.local_epoch, acc[-1], runtime),
        'Global Accuracy',
        'Round',
        "acc"
    )
    save_json(
        log,
        args.path_to_save + 'log_{}_{}_{}_R[{}]_l_[{}]_bs_[{}]_le[{}]_acc[{:.4f}]_time[{:.3f}].json'.format(args.dataset, args.exp_id, args.selection, args.global_epoch, args.l, args.selection_size, args.local_epoch, acc[-1], runtime)
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your script")
    # Add arguments here:
    # overall settings
    parser.add_argument('--exp_id', type=str, default='003', help='experiment id')
    parser.add_argument('--path_to_save', type=str, default='result/', help='path to save results')
    parser.add_argument('--seed', type=int, default=24, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:2', help='device to save tensors')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset, e.g., cifar10 / fashion')
    parser.add_argument('--n_clients', type=int, default=100, help='number of clients')

    # local train
    parser.add_argument('--local_epoch', type=int, default=5, help='local training epoch in each communication round')

    # server
    parser.add_argument('--global_epoch', type=int, default=1000, help='number of communication round')
    parser.add_argument('--cluster', type=str, default='edgeflow', help='method of cluster forming (edgeflow / seqflow / rand)')
    parser.add_argument('--cluster_size', type=int, default=10, help='size of a cluster')
    parser.add_argument('--l', type=int, default=1, help='the server will aggregate the cluster_size // l smallest lambda clients')

    # dataset
    parser.add_argument('--r0', type=float, default=0.95, help='the non-iid extent of local data')
    parser.add_argument('--r1', type=float, default=0.98, help='the non-iid extent of local data')
    parser.add_argument('--p', type=str, default='[0.1/0.0/0.9]', help='fraction of clients of iid , r0-noniid, r1-noniid')
    args = parser.parse_args()
    main(args)