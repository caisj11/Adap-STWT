# -*- coding: utf-8 -*-
# @Time : 2023/8/5 16:02
# @Author : Caisj
import math
import os.path
import torch
from torch.utils.data import DataLoader
import config
from model.AdapSTWT.Adap_STWT import Models
from model.AdapSTWT.graph_learn import GraphLearn
from preprocess.datasets import geometric_dataset
from trainer.AdapSTWT_trainer import AdapSTWTTrainer
from utils.scaler import StandardScaler
import logging


def main():
    torch.manual_seed(config.AdapSTWT_config_QD['seed'])
    torch.cuda.manual_seed(config.AdapSTWT_config_QD['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("INFO: device = ", device)
    # read  data and model config
    data_config, model_config = config.QD_config, config.AdapSTWT_config_QD
    # read dataset, standardization
    data_path = './data/flow/QD'
    data_scaler = StandardScaler(axis=(0, 1, 2))
    data_names, traj_names, time_names = ('train_flow.npy', 'val_flow.npy', 'test_flow.npy'), ('train_traj.npy', 'val_traj.npy', 'test_traj.npy'), ('train_time.npy', 'val_time.npy', 'test_time.npy')
    data_loaders = []
    for data_name, traj_name, time_name in zip(data_names, traj_names, time_names):
        dataset = geometric_dataset(data_path=os.path.join(data_path, data_name), traj_path=os.path.join(data_path, traj_name),
                                    time_path=os.path.join(data_path, time_name),
                                    node_num=data_config['node_num'], hist_num=data_config['time_step'], pred_num=data_config['pred_step'])
        if data_name == 'train_flow.npy':
            data_scaler.fit(dataset.data['x'])
        dataset.fit(data_scaler)
        data_loader = DataLoader(dataset, batch_size=model_config['batch_size'])
        data_loaders.append(data_loader)

    model_pred = Models(e_layers=model_config['encoder_num'], d_layers=model_config['decoder_num'],
                        node_num=data_config['node_num'], time_step=data_config['time_step'],
                        pre_num=data_config['pred_step'], graph_dim=model_config['graph_dim'],
                        d_model=model_config['time_dim'], heads=model_config['heads'],
                        dropout=model_config['dropout'], device=device).to(device)

    model_graph = GraphLearn(num_nodes=data_config['node_num'], time_step=data_config['time_step'], heads=model_config['heads'],
                             dropout=model_config['dropout'], graph_emb=model_config['graph_embedding'], device=device).to(device)

    # loss function
    optimizer_pred = torch.optim.Adam(model_pred.parameters(), lr=config.AdapSTWT_config_QD['learning_rate'], weight_decay=config.AdapSTWT_config_QD['weight_decay'])
    optimizer_graph = torch.optim.Adam(model_graph.parameters(), lr=config.AdapSTWT_config_QD['graph_learning_rate'])
    # learning rate decay
    scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pred, T_max=50, eta_min=1e-6)
    # scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_pred, T_0=10, T_mult=2)
    scheduler_graph = torch.optim.lr_scheduler.StepLR(optimizer_graph, step_size=30, gamma=0.9)
    
    # ------------------------ train ---------------------
    model_trainer = AdapSTWTTrainer(model_pred=model_pred, model_graph=model_graph, optimizer_pred=optimizer_pred,
                                   optimizer_graph=optimizer_graph, scheduler_pred=scheduler_pred, scheduler_graph=scheduler_graph,
                                   epoch_num=model_config['epoch'], num_iter=model_config['num_iter'], max_adj_num=model_config['max_adj_num'],
                                   scaler=data_scaler, model_save_path=model_config['model_save_path'], data_config=data_config, model_config=model_config)

    model_trainer.train(data_loaders[0], data_loaders[1])
    model_trainer.test(data_loaders[-1])
    



if __name__ == '__main__':
    # run model
    main()