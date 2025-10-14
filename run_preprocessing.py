# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/20 23:06
# @Author : caisj
from my_logging import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import config
from config import Storage_Traffic_Flow_Path_Name, Storage_Taxi_Flow_Path_Name, Storage_Taxi_Speed_Path_Name, \
    Storage_CD_Taxi_Flow_Path_Name
from preprocess.preprocessing_data import SensorsTrafficConversion, TaxiTrafficConversion, ChengDuTaxiTrafficConversion
import warnings

from utils.get_graph import get_drive_distance_graph, get_DTW_graph, traj_word2vec, get_CD_trajectory, \
    get_QD_trajectory, get_spherical_distance
from utils.tools import data_smooth, drop_small_flow
warnings.filterwarnings("ignore")


def storage_traffic_flow_speed(path_and_name):
    # Collect traffic flow and save to the data folder: sensor flow has no speed
    STC = SensorsTrafficConversion()
    all_traffic_flow = STC.get_all_traffic_flow()
    all_traffic_flow.to_csv(path_and_name, index=False)


def storage_taxi_flow_speed(path_and_name_flow, path_and_name_speed):
    # Collect taxi flow and average speed
    TTC = TaxiTrafficConversion()
    all_taxi_flow, all_taxi_speed = TTC.get_all_taxi_flow_speed()
    # Save data
    all_taxi_flow.to_csv(path_and_name_flow, index=False)
    all_taxi_speed.to_csv(path_and_name_speed, index=False)


def storage_CD_taxi_flow(path_and_name):
    # Collect Chengdu traffic flow and save to the data folder
    CDTTC = ChengDuTaxiTrafficConversion()
    all_traffic_flow = CDTTC.get_all_traffic_flow()
    all_traffic_flow.to_csv(path_and_name, index=False)


def flow_process(data, flow_threshold, step, city):
    # Kalman smoothing
    for column_name in tqdm(data.columns.tolist(), desc='Kalman smoothing'):
        if column_name.isdigit():
            data[column_name] = data_smooth(data[column_name], step)
    # Remove sensors with average flow below the specified threshold
    data = drop_small_flow(data, flow_threshold)
    if city == 'QD':
        # Remove invalid sensors: only keep sensors that have coordinates
        df_location = pd.read_excel(r'./data/node_location/QD_nodes.xlsx')
        location_map = {}
        for index, row in df_location.iterrows():
            location_map[str(row['crossroadID'])] = [row['lng'], row['lat']]
        for name in data.columns.tolist():
            if name.isdigit() and name not in location_map:
                del data[name]
    return data


def time_embedding():
    '''
    Get time embedding
    :return: [T,N,C]
    '''
    from datetime import datetime, timedelta

    node_num = 134

    time_embedding = []
    for i in range(1, 15):
        # start time
        start_date = datetime(2019, 8, i, 7, 5)
        end_date = datetime(2019, 8, i, 19, 0)

        # define time interval
        interval = timedelta(minutes=5)

        # generate time intervals
        time_intervals = []
        current_date = start_date
        while current_date <= end_date:
            time_intervals.append(current_date)
            current_date += interval

        # extract time features and normalize
        # day = [i.day / 30.0 - 0.5 for i in time_intervals]
        hour = [i.hour / 23.0 - 0.5 for i in time_intervals]
        dayofweek = [i.weekday / 6.0 - 0.5 for i in time_intervals]

        # day = np.array([day for _ in range(node_num)]).T # [144,N]
        hour = np.array([hour for _ in range(node_num)]).T # [144,N]
        dayofweek = np.array([dayofweek for _ in range(node_num)]).T # [144,N]
        embedding = np.stack((hour, dayofweek), axis=-1) # [144,N,2]

        # sliding window sampling
        for k in range(embedding.shape[0] - 12 - 12 + 1):
            time_slice = embedding[k:k+24, :, :]
            time_embedding.append(time_slice)
    time_embedding = np.array(time_embedding)
    return time_embedding


def save_QD_prior_graph(df_flow, key):
    # ------------------ step1 ------------------------
    # reading nodes location
    df_location = pd.read_excel(r'./data/node_location/QD_nodes.xlsx')
    location_map = {}
    for index, row in df_location.iterrows():
        location_map[str(row['crossroadID'])] = [row['lng'], row['lat']]
    # distance_matrix = get_drive_distance_graph(df_flow, location_map, key)
    # np.save("./data/graph/QD/distance_matrix.npy", distance_matrix)
    distance_matrix = get_spherical_distance(df_flow, location_map)
    np.save("./data/graph/QD/spherical_distance_matrix.npy", distance_matrix)

    # ------------------ step2 ------------------------
    # # DTW graph
    # dtw_matrix = get_DTW_graph(df_flow)
    # np.save("./data/graph/QD/dtw_matrix.npy", dtw_matrix)

    # ------------------ step3 ------------------------
    # save history trajectory
    logger.info('word embedding...')
    geohash_seq = get_QD_trajectory()
    assert len(geohash_seq) == 121 * 19
    train_set = geohash_seq[:1694]
    val_set = geohash_seq[1694:1936]
    test_set = geohash_seq[1936:]
    with open('data/flow/QD/train_traj.json', 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open('data/flow/QD/val_traj.json', 'w', encoding='utf-8') as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    with open('data/flow/QD/test_traj.json', 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    
def save_CD_prior_graph(df_flow, key):
    nodes_id = [name for name in df_flow.columns.tolist() if name.isdigit()]
    np.save("./data/graph/CD/nodes_id.npy", nodes_id)
    # ------------------ step1 ------------------------
    df_location = pd.read_csv(r'./data/node_location/CD_railway.csv')
    location_map = {}
    for index, row in df_location.iterrows():
        location_map[str(row['node_id'])] = [row['lng'], row['lat']]
    # distance_matrix = get_drive_distance_graph(df_flow, location_map, key)
    # np.save("./data/graph/CD/distance_matrix.npy", distance_matrix)
    distance_matrix = get_spherical_distance(df_flow, location_map)
    np.save("./data/graph/CD/spherical_distance_matrix.npy", distance_matrix)
    
    # ------------------ step2 ------------------------
    # DTW graph
    # dtw_matrix = get_DTW_graph(df_flow)
    # np.save("./data/graph/CD/dtw_matrix.npy", dtw_matrix)

    # ------------------ step3 ------------------------
    transfer_matrix = get_CD_trajectory()
    np.save('./data/trajectory/CD_transfer_matrix.npy', transfer_matrix)
    
def sliding_window_sampling(data, window_size=24, samples_per_day=144):
    total_samples, num_features = data.shape
    total_days = total_samples // samples_per_day
    windows_per_day = samples_per_day - window_size + 1
    total_windows = total_days * windows_per_day
    
    windowed_data = np.zeros((total_windows, window_size, num_features))
    
    window_idx = 0
    for day in range(total_days):
        day_start = day * samples_per_day
        day_end = (day + 1) * samples_per_day
        day_data = data[day_start:day_end]
        
        for i in range(windows_per_day):
            window_data = day_data[i:i + window_size]
            windowed_data[window_idx] = window_data
            window_idx += 1
    
    return windowed_data


if __name__ == "__main__":
    # ----------------  step1: compute Qingdao sensor traffic flow -------------
    # Convert sensor traffic flow for August and September
    storage_traffic_flow_speed(path_and_name=Storage_Traffic_Flow_Path_Name)

    #----------------  step2: compute Chengdu subway taxi flow -------------
    # Convert Chengdu taxi data
    storage_CD_taxi_flow(path_and_name=Storage_CD_Taxi_Flow_Path_Name)

    #----------------  step3: flow processing (temporal denoising, smoothing, sensor filtering) -------------
    logger.info('QD flow processing (temporal denoising, smoothing, sensor filtering)...')
    # Qingdao flow conversion
    df_flow_QD = pd.read_csv(r'data/train_flow/QD_traffic_flow.csv')
    df_flow_QD = df_flow_QD[df_flow_QD['month_day'] < '09-01'].copy() # September data is incorrect, remove
    df_flow_QD = flow_process(df_flow_QD, config.QD_Flow_Threshold, 144, city='QD')
    df_flow_QD.to_csv(r'data/train_flow/QD_flow.csv', index=False)
    features = df_flow_QD.iloc[:, 1:].values
    train_data = features[:14*144]
    val_data = features[14*144:16*144]
    test_data = features[16*144:]
    train_windowed = sliding_window_sampling(train_data)
    val_windowed = sliding_window_sampling(val_data)
    test_windowed = sliding_window_sampling(test_data)
    np.save(r'data/flow/QD/train_flow.npy', train_windowed)
    np.save(r'data/flow/QD/val_flow.npy', val_windowed)
    np.save(r'data/flow/QD/test_flow.npy', test_windowed)

    logger.info('CD flow processing (temporal denoising, smoothing, sensor filtering)...')
    # Chengdu flow conversion
    df_flow_CD = pd.read_csv(r'data/train_flow/CD_taxi_flow.csv')
    df_flow_CD = flow_process(df_flow_CD, config.CD_Flow_Threshold, 216, city='CD')
    df_flow_CD.to_csv(r'data/train_flow/CD_flow.csv', index=False)

    #----------------  step4: generate prior adjacency graphs  ----------------------
    logger.info('QD prior adjacency graph generation...')
    # Compute Qingdao adjacency graph
    key = config.key
    df_flow = pd.read_csv(r'./data/original_flow/QD_flow.csv')
    save_QD_prior_graph(df_flow, key)

    # Compute Chengdu adjacency graph
    key = config.key
    df_flow = pd.read_csv(r'./data/original_flow/CD_flow.csv')
    save_CD_prior_graph(df_flow, key)

