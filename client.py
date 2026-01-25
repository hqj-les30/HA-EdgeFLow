import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
import copy

class Client_Configs:
    def __init__(self, args):
        self.device = args.device
        self.local_epoch = args.local_epoch
        self.lr = 0.001
        self.optimizer = torch.optim.Adam
        self.criterion = nn.CrossEntropyLoss()

class Client:
    def __init__(self, id, ds, mix_mode, configs):
        self.ID = id
        self.ds = ds
        self.mix_mode = mix_mode
        self.configs = configs

        self.server = None
        self.model = None

        self.loader = DataLoader(self.ds, batch_size=64, shuffle=True)

    def attach_to_server(self, server):
        self.server = server

    def run(self):
        self.model = copy.deepcopy(self.server.glob_model)
        self.model.to(self.configs.device)
        self.model.eval()
        # loss_est = self.__loss_estimate()
        grad_est, loss_est = self.__gradient_estimate()
        self.model.train()
        loss = self.__local_train()
        self.model.to('cpu')
        self.model.eval()
        return grad_est, loss

    def __local_train(self):
        optimizer = self.configs.optimizer(self.model.parameters(), lr = self.configs.lr)
        avg_loss = 0.
        for _ in range(self.configs.local_epoch):
            epoch_loss = 0.
            for x, y in self.loader:
                x = x.to(self.configs.device)
                y = y.to(self.configs.device)
                out = self.model(x)
                loss = self.configs.criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            avg_loss += epoch_loss
        avg_loss /= self.configs.local_epoch
        return avg_loss

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict())
    
    def __loss_estimate(self):
        avg_loss = 0.
        total_samples = 0
        for x, y in self.loader:
            x = x.to(self.configs.device)
            y = y.to(self.configs.device)
            out = self.model(x)
            loss = self.configs.criterion(out, y)
            avg_loss += loss.item()
            total_samples += x.shape[0]
        return avg_loss / total_samples
    
    def __gradient_estimate(self):
        avg_loss = 0.
        total_samples = 0   
        grad_list = []
        for x, y in self.loader:
            x = x.to(self.configs.device)
            y = y.to(self.configs.device)
            out = self.model(x)
            loss = self.configs.criterion(out, y)
            grad = torch.autograd.grad(loss, self.model.parameters())
            grad_list.append(grad)
            avg_loss += loss.item()
            total_samples += x.shape[0]
        
        avg_grad = []
        for i in range(len(grad_list[0])):
            layer_grads = torch.stack([g[i] for g in grad_list])
            avg_grad.append(torch.mean(layer_grads, dim=0))
        return avg_grad, avg_loss / total_samples
    
    

def client_preparation(local_datasets, mode_vec, args):
    configs = Client_Configs(args)
    client_vec = []
    for i, dataset in local_datasets.items():
        client_vec.append(
            Client(i, dataset, mode_vec[i], configs)
        )
    return client_vec