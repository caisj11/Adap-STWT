# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/18 23:17
# @Author : caisj

# --------------------------------  Qingdao preprocessing ------------------------------
import pandas as pd


Train_Traffic_Path = r"E:/数据集汇总/青岛数据集/trainCrossroadFlow"
Train_Taxi_Path = r"E:/数据集汇总/青岛数据集/trainTaxiGPS"
Test_Traffic_Path = r"E:/数据集汇总/青岛数据集/testCrossroadFlow"
Test_Taxi_Path = r"E:/数据集汇总/青岛数据集/testTaxiGPS"
Crossroad_Location_Path_Name = r"E:/数据集汇总/青岛数据集/crossroad_location_wgs.xlsx"
Storage_Traffic_Flow_Path_Name = r"data/train_flow/QD_traffic_flow.csv"
Storage_Taxi_Flow_Path_Name = r"./data/train_taxi/all_taxi_flow.csv"
Storage_Taxi_Speed_Path_Name = r"./data/train_taxi/all_taxi_speed.csv"
QD_Flow_Threshold = 20
QD_Step = 144
Crossroad_ID = []

# --------------------------------  Chengdu preprocessing -------------------------------
CD_Crossroad_Location_Path_Name = r"data/node_location/CD_railway.csv"
CD_Taxi_Path = r'E:/数据集汇总/成都出租车数据集'
Storage_CD_Taxi_Flow_Path_Name = r'data/train_flow/CD_taxi_flow.csv'
CD_Flow_Threshold = 20
CD_Step = 216
Box_Width = 500
CD_Bounds = [103.91, 30.50, 104.17, 30.79]
QD_Bounds = [120.29, 36.05, 120.51, 36.22]

Train_Step = 12
Pred_Step = 12
CD_Dataset_Path = './data/flow/CD'
QD_Dataset_Path = './data/flow/QD'
split_ratio = [0.7, 0.1, 0.2]
Batch_Size = 128

# API Gaode
key = '39c2286b9f987ebbf4446be1ba5e5e96'


# ---------------------------------- data config ------------------------------------
QD_config = {
    'train': './data/flow/QD/train.npy',
    'val': './data/flow/QD/val.npy',
    'test': './data/flow/QD/test.npy',
    'dist_mx': './data/graph/QD/distance_matrix.npy',
    'dtw_mx': './data/graph/QD/dtw_matrix.npy',
    'adj_mx': './data/graph/QD/adjacency_matrix.npy',
    'word2vec_matrix': './data/graph/QD/word2vec_matrix.npy',
    'node_num': 134,
    'time_step': 12,
    'pred_step': 12,
}

CD_config = {
    'train': './data/flow/CD/train.npy',
    'val': './data/flow/CD/val.npy',
    'test': './data/flow/CD/test.npy',
    'dist_mx': './data/graph/CD/distance_matrix.npy',
    'dtw_mx': './data/graph/CD/dtw_matrix.npy',
    'adj_mx': './data/graph/CD/adjacency_matrix.npy',
    'word2vec_matrix': './data/graph/CD/word2vec_matrix.npy',
    'node_num': 149,
    'time_step': 12,
    'pred_step': 12,
}



# ---------------------------------- model config --------------------------------------
QD_data_config = {
    'step_num_in': 12,
    'step_num_out': 12,
    'num_nodes': 134,
    'input_size': 1,
    'batch_size': 32,
    'device': 'cuda:0',
}
CD_data_config = {
    'step_num_in': 12,
    'step_num_out': 12,
    'num_nodes': 149,
    'input_size': 1,
    'batch_size': 128,
    'device': 'cuda:1',
}

AdapSTWT_config_QD = {
    'seed': 2023,
    'encoder_num': 2,
    'decoder_num': 1,
    'conv_dim': 32,
    'graph_dim': 32,
    'time_dim': 32,
    'heads': 4,
    'dropout': 0,
    'graph_embedding': 32,
    'learning_rate': 0.001,
    'graph_learning_rate': 0.001,
    'weight_decay': 0.0001,
    'epoch': 5,
    'num_iter': 200,
    'max_adj_num': 3,
    'batch_size': 64,
    'model_save_path': './model_states/QD/Adap_STWT.pkl',
    'teacher_forcing_ratio': 0.4,
}

AdapSTWT_config_CD = {
    'seed': 2023,
    'encoder_num': 2,
    'decoder_num': 1,
    'conv_dim': 32,
    'graph_dim': 32,
    'time_dim': 32,
    'heads': 4,
    'dropout': 0,
    'graph_embedding': 32,
    'learning_rate': 0.001,
    'graph_learning_rate': 0.001,
    'weight_decay': 0.0001,
    'epoch': 5,
    'num_iter': 200,
    'max_adj_num': 3,
    'batch_size': 64,
    'model_save_path': './model_states/CD/Adap_STWT.pkl',
    'teacher_forcing_ratio': 0.4,
}