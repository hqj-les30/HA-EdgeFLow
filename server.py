import copy

import torch
import numpy as np
import torch.utils
import torch.utils.data

import time

class Server_Configs:
    def __init__(self, args):
        self.device = args.device
        self.global_epoch = args.global_epoch
        self.model_fn = args.model_fn
        self.client_sampler = args.client_sampler
        self.select_size = args.selection_size
        self.n_class = args.n_class
        self.lambda_ratio = args.l
        self.criterion = torch.nn.CrossEntropyLoss()

class Server_Logger:
    def __init__(self, test_datset):
        self.client_selelction = []
        self.client_loss = []
        self.average_loss = []
        self.global_accuracy = []
        self.global_accuracy_per_class = []

        self.ds = test_datset
        self.loader = torch.utils.data.DataLoader(
            self.ds, batch_size=200, shuffle=False
        )

        self.log = {}

        self.start_time = None
        self.end_time = None

    def run(self, t, device, global_model, client_index, client_loss, n_class = 10):
        info = ''
        # log = {}
        average_loss = sum(client_loss) / len(client_loss)
        self.average_loss.append(average_loss)
        self.client_selelction.append(client_index)
        self.client_loss.append(client_loss)
        acc, acc_per_class = self.__validate(device, global_model, n_class)
        self.global_accuracy.append(acc)
        self.global_accuracy_per_class.append(acc_per_class)
        current_time = time.time()
        self.log[str(t)] = {
            'client selected': client_index,
            'client avergae loss': average_loss,
            'global accuracy': acc,
            'class accuracy': acc_per_class,
            'time': round(current_time - self.start_time, 3)
        }

        print('Global Round: {}, Loss: {:.3f}, Accuracy: {:.3f}'.format(t, average_loss, acc))
    
    def __validate(self, device, model, n_class):
        model.eval()
        model.to(device)

        correct = 0
        total = 0
        correct_per_class = [0 for _ in range(n_class)]
        total_per_class = [0 for _ in range(n_class)]

        with torch.no_grad():
            for x, labels in self.loader:

                x, labels = x.to(device), labels.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)

                for label, pred in zip(labels, preds):
                    total += 1
                    total_per_class[label.item()] += 1
                    if pred.item() == label.item():
                        correct += 1
                        correct_per_class[label.item()] += 1

        overall_acc = correct / total
        per_class_acc = [correct_per_class[cls] / total_per_class[cls] for cls in range(n_class)]

        model.to('cpu')
        return overall_acc, per_class_acc

        

class Server:
    def __init__(self, client_vec, mix_mode_vec, configs: Server_Configs, logger: Server_Logger, exp_id='default_exp', proxy_dataset=None):
        self.exp_id = exp_id
        self.configs = configs
        self.logger = logger
        
        self.glob_model = self.configs.model_fn(n_class=self.configs.n_class)
        
        self.client_vec = client_vec
        self.n_clients = len(client_vec)
        self.mix_mode_vec = mix_mode_vec

        self.proxy_dataset = proxy_dataset
        if self.proxy_dataset is not None:
            self.proxy_loader = torch.utils.data.DataLoader(self.proxy_dataset, batch_size=200, shuffle=False)
        self.__zero_weights = {}
        for key, val in self.glob_model.state_dict().items():
            self.__zero_weights[key] = torch.zeros(size=val.shape, dtype=torch.float32)

        self.client_sampler = self.configs.client_sampler(self.n_clients, mix_mode_vec, self.configs.select_size, self.configs.global_epoch)

    def run(self):
        for c in self.client_vec:
            c.attach_to_server(self)
        self.logger.start_time = time.time()
        for t in range(self.configs.global_epoch):
            client_selected = next(self.client_sampler)
            cluster_lambda = []

            F_mt, client_loss = [], []
            for k in client_selected:
                F_c, loss = self.client_vec[k].run()
                client_loss.append(loss)
                F_mt.append(F_c)

            if self.proxy_dataset is not None:
                self.glob_model.to(self.configs.device)
                proxy_grad = self.__run_proxy_gradient()
                self.glob_model.to('cpu')

            grad_dist = []
            for client_grad in F_mt:
                total_dist = 0
                for c_layer, p_layer in zip(client_grad, proxy_grad):
                    total_dist += torch.norm(c_layer - p_layer).item()
                grad_dist.append(total_dist)
            
            loss_rank = np.argsort(grad_dist)[:len(client_selected)//self.configs.lambda_ratio]
            agg_selected = [client_selected[k] for k in loss_rank]
            self.__run_aggregation(agg_selected)
            # self.__run_aggregation(client_selected)
            self.logger.run(t+1, self.configs.device, self.glob_model, client_selected, client_loss, self.configs.n_class)
        self.logger.end_time = time.time()

    def __run_aggregation(self, client_selected):
        new_weights = copy.deepcopy(self.__zero_weights)
        for k in client_selected:
            w = self.client_vec[k].get_model_params()
            for key in new_weights.keys():
                new_weights[key] += (1.0 / len(client_selected)) * w[key]
        
        self.glob_model.load_state_dict(new_weights)

    def __run_proxy_gradient(self):
        proxy_grad = []
        for x, y in self.proxy_loader:
            x = x.to(self.configs.device)
            y = y.to(self.configs.device)
            out = self.glob_model(x)
            loss = self.configs.criterion(out, y)
            grad = torch.autograd.grad(loss, self.glob_model.parameters())
            proxy_grad.append(grad)
        
        avg_grad = []
        for i in range(len(proxy_grad[0])):
            layer_grads = torch.stack([g[i] for g in proxy_grad])
            avg_grad.append(torch.mean(layer_grads, dim=0))
        return avg_grad
            



