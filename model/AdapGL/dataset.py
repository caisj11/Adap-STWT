# -*- coding: utf-8 -*-
# @Time : 2023/9/1 17:16
# @Author : Caisj
from torch_geometric.data import Data, Dataset
import numpy as np
import torch


class geometric_dataset(Dataset):
    """
    data_path:flow流量数据路径
    traj_path:节点的实时状态转移概率矩阵路径
    node_num: 交通节点数量
    hist_num：历史序列长度（前多少条数据训练）
    pred_num：预测不长
    """
    def __init__(self, data_path, hist_num=12, pred_num=12):
        self.hist_num = hist_num
        self.pred_num = pred_num
        # 加载数据
        print("--- 开始加载流量数据 ---")
        data = np.load(data_path)
        self.data = {'x': data[:, :hist_num, :], 'y': data[:, hist_num:, :]}

    def fit(self, scaler):
        self.data['x'] = scaler.transform(self.data['x'])
        self.data['y'] = scaler.transform(self.data['y'], axis=0)

    def __getitem__(self, index):
        # 读取流量数据
        x_i = torch.tensor(self.data['x'][index, :, :], dtype=torch.float).unsqueeze(2) #[T,N,C]
        y_i = torch.tensor(self.data['y'][index, :, :], dtype=torch.float) #[T,N]
        return [x_i, y_i]

    def __len__(self):
        return self.data['x'].shape[0]