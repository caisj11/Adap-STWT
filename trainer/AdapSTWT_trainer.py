'''
Author: Caisj
Date: 2024-07-11 19:37:32
LastEditTime: 2025-10-13 18:00:16
'''
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from .base import Trainer, TFTrainer


class AdapSTWTTrainer(Trainer):
    def __init__(self, model_pred, model_graph, optimizer_pred, optimizer_graph, scheduler_pred,
                 scheduler_graph, epoch_num, num_iter, max_adj_num, scaler, model_save_path, data_config, model_config):
        self.model_pred: nn.Module = model_pred
        self.model_graph: nn.Module = model_graph
        self.num_iter: int = num_iter
        self.model_save_path: str = model_save_path
        self.device = next(self.model_pred.parameters()).device
        self.max_adj_num: int = max_adj_num
        self.model_config = model_config
        self.node_num = data_config['node_num']
        self.time_step = data_config['time_step']
        adj_mx_list = self.add_adj_mx_list(data_config)
        self.adj_mx_list = [[adj_mx, -1] for adj_mx in adj_mx_list]

        self.epsilon = 1 / 134 * 0.5
        self.best_adj_mx = None
        self.update_best_adj_mx('union', threshold=False)

        model_save_dir, model_name = os.path.split(self.model_save_path)
        self.graph_save_path = os.path.join(model_save_dir, 'GRAPH.pkl')

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        self.model_pred_trainer = self.ModelPredTrainer(model_pred, optimizer_pred, scheduler_pred, epoch_num, scaler, model_save_path, self)

        self.model_graph_trainer = self.GraphLearnTrainer(model_graph, optimizer_graph, scheduler_graph, epoch_num, scaler, self.graph_save_path, self)

        best_save_dir = os.path.join(model_save_dir, model_name.split('.')[0])
        self.best_pred_path = os.path.join(best_save_dir, model_name)
        self.best_graph_path = os.path.join(best_save_dir, 'best_adj_mx.npy')

        if not os.path.exists(best_save_dir):
            os.mkdir(best_save_dir)

    def add_adj_mx_list(self, data_config):
        fusion_adj_mx = []
        for name in ['adj_mx']:
            adj_mx = np.load(data_config[name])
            # adj_mx = np.where(adj_mx > 0.6, adj_mx, 0)
            adj_mx = torch.tensor(adj_mx, dtype=torch.float32, device=self.device)
            fusion_adj_mx.append(adj_mx)
        return fusion_adj_mx

    def update_adj_mx_list(self, data_loader, new_adj_mx):
        # delete the graph with the maximum loss during each update, 
        # then add the current dynamic graph to update self.adj_mx_list
        self.adj_mx_list.append([new_adj_mx, 0])
        max_loss, max_index = torch.finfo(torch.float32).min, -1

        for i, (adj_mx, _) in enumerate(self.adj_mx_list):
            cur_loss = self.evaluate(data_loader, adj_mx)
            self.adj_mx_list[i][-1] = cur_loss
            if cur_loss > max_loss:
                max_loss, max_index = cur_loss, i

        if len(self.adj_mx_list) > self.max_adj_num:
            self.adj_mx_list.pop(max_index)

    def update_best_adj_mx(self, criteria, threshold=True):
        """
        Update self.best_adj_mx.

        criteria:
            - replace: use the newest subgraph as best_adj_mx;
            - union: combine the adj_mx in self.adj_mx_list;
            - weight_union: weighted sum of adj_mx in self.adj_mx_list according to
                evaluate loss.
            - channel_attention: channel attention is used to fuse self.adj_mx_list
        """
        if criteria == 'replace':
            best_adj_mx = self.adj_mx_list[-1][0]

        elif criteria == 'union':
            adj_mx_sum = torch.zeros_like(self.adj_mx_list[0][0])
            adj_num_sum = torch.zeros_like(adj_mx_sum)
            for adj_mx, _ in self.adj_mx_list:
                adj_mx_sum += adj_mx
                adj_num_sum += (1 + torch.sign(adj_mx - 1e-4)) / 2
            adj_mx_sum /= adj_num_sum
            adj_mx_sum[torch.logical_or(torch.isnan(adj_mx_sum), torch.isinf(adj_mx_sum))] = 0
            best_adj_mx = adj_mx_sum

        else:
            loss_tensor = torch.tensor([x[-1] for x in self.adj_mx_list], requires_grad=False)
            loss_weight = F.softmax(loss_tensor.max() - loss_tensor, dim=0)

            best_adj_mx = torch.zeros_like(self.adj_mx_list[0][0])
            for i, (adj_mx, _) in enumerate(self.adj_mx_list):
                best_adj_mx += loss_weight[i] * adj_mx

        if threshold:
            d = best_adj_mx.sum(dim=-1) ** (-0.5)
            best_adj_mx = torch.relu(d.view(-1, 1) * best_adj_mx * d - self.epsilon)
        self.best_adj_mx = best_adj_mx

    def update_num_epoch(self, cur_iter):
        if cur_iter == self.num_iter // 2 + 1:
            self.model_pred_trainer.max_epoch_num += 5

    def train_one_epoch(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape', 'r2'), cur_iter=None):
        # step 1: training spatio-temporal model
        self.model_pred_trainer.train(train_data_loader, eval_data_loader, metrics)
        self.model_pred.load_state_dict(torch.load(self.model_save_path))
        # step 2: learning new graph
        self.model_graph_trainer.train(train_data_loader, eval_data_loader, metrics)
        self.model_graph.load_state_dict(torch.load(self.graph_save_path))
        # Add the learned graph to self.adj_mx_list, update self.best_adj_mx.
        new_adj_mx = self.model_graph_trainer.generate_graph(self.model_graph, train_data_loader, self.best_adj_mx).detach()
        # step 3: update best graph structure
        print('Evaluation results of all subgraphs:')
        self.update_adj_mx_list(eval_data_loader, new_adj_mx)
        if cur_iter > self.num_iter * 0.8:
            self.update_best_adj_mx('replace')
        else:
            self.update_best_adj_mx('weight_union')

    @torch.no_grad()
    def evaluate(self, data_loader, adj_mx):
        """
        Test the prediction loss on data set 'data_loader' using 'adj_mx'.
        """
        loss, _, _ = self.model_pred_trainer.evaluate(data_loader, adj_mx=adj_mx)
        return loss

    @torch.no_grad()
    def test(self, data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
        self.model_pred.load_state_dict(torch.load(self.best_pred_path))
        best_adj_mx_np = np.load(self.best_graph_path)
        best_adj_mx = torch.tensor(
            data=best_adj_mx_np,
            dtype=torch.float32,
            device=self.device
        )
        sparsity = (best_adj_mx_np == 0).sum() / (best_adj_mx_np.shape[0] ** 2)
        print('Sparsity: {:.4f}'.format(sparsity))
        print('Test results of current graph: ')
        _, y_true, y_pred = self.model_pred_trainer.evaluate(data_loader, metrics, adj_mx=best_adj_mx)
        # np.save('perdiction/QD/y_pred.npy', y_pred)
        # np.save('perdiction/QD/y_true.npy', y_true)
        self.model_pred_trainer.print_test_result(y_pred, y_true, metrics)

    def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
        print('Start Training...')
        min_loss = torch.finfo(torch.float32).max
        for i in range(self.num_iter):
            print('Iteration {}:'.format(i + 1))
            self.train_one_epoch(train_data_loader, eval_data_loader, metrics, i)
            print('Evaluation results of current graph:')
            cur_loss = self.evaluate(eval_data_loader, self.best_adj_mx)
            if cur_loss < min_loss:
                copyfile(self.model_save_path, self.best_pred_path)
                np.save(self.best_graph_path, self.best_adj_mx.cpu().numpy())
                min_loss = cur_loss
            self.update_num_epoch(i + 1)

    class ModelPredTrainer(TFTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path)
            self.outer_obj: AdapSTWTTrainer = outer_obj
            self.num_iter: int = 1
            self.batches_seen: int = 0

        def train_one_epoch(self, data_loader):
            self.model.train()
            for _ in range(self.num_iter):
                for x_enc, x_enc_mark, x_dec, x_dec_mark, traj_mx, y in data_loader:
                    x_enc = x_enc.type(torch.float32).to(self.device)
                    x_enc_mark = x_enc_mark.type(torch.float32).to(self.device)
                    x_dec = x_dec.type(torch.float32).to(self.device)
                    x_dec_mark = x_dec_mark.type(torch.float32).to(self.device)
                    y = y.type(torch.float32).to(self.device)
                    pred = self.model(x_enc, x_enc_mark, x_dec, x_dec_mark, adj_mx=self.outer_obj.best_adj_mx)
                    loss = F.l1_loss(pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.batches_seen += 1

        def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
            print('Round for prediction model:')
            super().train(train_data_loader, eval_data_loader, metrics)

        @torch.no_grad()
        def evaluate(self, data_loader, metrics=('mae', 'rmse', 'mape', 'r2'), adj_mx=None):
            if adj_mx is None:
                adj_mx = self.outer_obj.best_adj_mx
            return super().evaluate(data_loader, metrics, adj_mx=adj_mx)

        @torch.no_grad()
        def test(self, data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
            self.model.load_state_dict(torch.load(self.model_save_path))
            _, y_true, y_pred = self.evaluate(data_loader, metrics)
            self.print_test_result(y_pred, y_true, metrics)

        @staticmethod
        def model_loss_func(y_pred, y_true):
            return F.l1_loss(y_pred, y_true)

    class GraphLearnTrainer(TFTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path)
            self.outer_obj: AdapSTWTTrainer = outer_obj
            self.num_iter: int = 1
            self.delta: int = 0.05

        def train_one_epoch(self, data_loader):
            self.model.train()
            best_adj_mx = self.outer_obj.best_adj_mx
            for _ in range(self.num_iter):
                for x_enc, x_enc_mark, x_dec, x_dec_mark, traj_mx, y in data_loader:
                    x_enc = x_enc.type(torch.float32).to(self.device)
                    x_enc_mark = x_enc_mark.type(torch.float32).to(self.device)
                    x_dec = x_dec.type(torch.float32).to(self.device)
                    x_dec_mark = x_dec_mark.type(torch.float32).to(self.device)
                    traj_mx = traj_mx.type(torch.float32).to(self.device)
                    y = y.type(torch.float32).to(self.device)
                    # generate new graph
                    adj_mx = self.model(x_enc, best_adj_mx, traj_mx)
                    # calculate the loss
                    pred = self.outer_obj.model_pred(x_enc, x_enc_mark, x_dec, x_dec_mark, adj_mx=adj_mx)
                    loss = F.l1_loss(pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
            print('Round for graph learning:')
            super().train(train_data_loader, eval_data_loader, metrics)

        @torch.no_grad()
        def evaluate(self, data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
            self.model.eval()
            adj_mx = torch.zeros_like(self.outer_obj.best_adj_mx)
            num = 0
            for x_enc, x_enc_mark, x_dec, x_dec_mark, traj_mx, y in data_loader:
                x_enc = x_enc.type(torch.float32).to(self.device)
                traj_mx = traj_mx.type(torch.float32).to(self.device)
                new_adj_mx = self.model(x_enc, self.outer_obj.best_adj_mx, traj_mx).detach()
                adj_mx += new_adj_mx
                num += 1
            adj_mx = adj_mx / num
            return self.outer_obj.model_pred_trainer.evaluate(data_loader, metrics, adj_mx=adj_mx)

        @torch.no_grad()
        def test(self, data_loader, metrics=('mae', 'rmse', 'mape', 'r2')):
            self.model.load_state_dict(torch.load(self.model_save_path))
            _, y_true, y_pred = self.evaluate(data_loader, metrics)
            self.print_test_result(y_pred, y_true, metrics)

        @torch.no_grad()
        def model_loss_func(self, x, traj, y_pred, y_true):
            """Loss function of Graph Learn Model."""
            mx_p = self.outer_obj.best_adj_mx
            mx_q = self.model(x, mx_p, traj)
            mx_delta = torch.sign(mx_q) - torch.sign(mx_p)
            sim_loss = F.relu(F.relu(mx_delta).mean() - self.delta) / self.delta
            pred_loss = F.l1_loss(y_pred, y_true)
            return pred_loss + sim_loss

        @torch.no_grad()
        def generate_graph(self, model, data_loader, adj_mx):
            best_adj_mx = torch.zeros_like(adj_mx)
            num = 0
            for x_enc, _, _, _, traj_mx, _ in data_loader:
                x_enc = x_enc.type(torch.float32).to(self.device)
                traj_mx = traj_mx.type(torch.float32).to(self.device)
                new_adj_mx = model(x_enc, adj_mx, traj_mx)
                best_adj_mx += new_adj_mx
                num += 1
            best_adj_mx /= num
            return best_adj_mx
