# -*- coding: utf-8 -*-
# @Time : 2023/6/20 19:45
# @Author : Caisj
from datetime import datetime
from scipy import spatial
import requests
import transbigdata
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import config
from my_logging import logger
from preprocess.preprocessing_data import ChengDuTaxiTrafficConversion, SensorsTrafficConversion, \
    multiprocessing_change_coordinate
from utils.map_matching import hmm_map_matching
from utils.multiprocessing import run_task_multiprocessing
from itertools import groupby
import osmnx as ox
from utils.tools import get_distance_hav, hmm_matching, read_openstreetmap


def get_spherical_distance(data, location_map):
    nodes_id = [name for name in data.columns.tolist() if name.isdigit()]
    distance_matrix = np.zeros((len(nodes_id), len(nodes_id)))
    for i in tqdm(range(len(nodes_id))):
        for j in range(i, len(nodes_id)):
            if i == j:
                distance = 0
            else:
                begin_lng, begin_lat, end_lng, end_lat = location_map[nodes_id[i]][0], location_map[nodes_id[i]][1], \
                location_map[nodes_id[j]][0], location_map[nodes_id[j]][1]
                distance = get_distance_hav(begin_lat, begin_lng, end_lat, end_lng)
            distance_matrix[i, j] = distance
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance_matrix[j, i] = distance_matrix[i, j]
    # std = np.std(distance_matrix)
    # for i in range(len(distance_matrix)):
    #         distance_matrix[i] = [math.exp(-dist / std) if dist < 0.2 else 0 for dist in distance_matrix[i]]
    return distance_matrix


def get_drive_distance_graph(data, location_map, key):
    nodes_id = [name for name in data.columns.tolist() if name.isdigit()]
    distance_matrix = np.zeros((len(nodes_id), len(nodes_id)))
    for i in tqdm(range(len(nodes_id))):
        for j in range(i, len(nodes_id)):
            begin_lng, begin_lat, end_lng, end_lat = location_map[nodes_id[i]][0], location_map[nodes_id[i]][1], \
            location_map[nodes_id[j]][0], location_map[nodes_id[j]][1]
            url = f'https://restapi.amap.com/v3/direction/driving?origin={begin_lng},{begin_lat}&destination={end_lng},{end_lat}&extensions=all&key={key}'
            response = requests.post(url)
            if i == j:
                distance = 0
            else:
                distance = response.json()['route']['paths'][0]['distance']
            distance_matrix[i, j] = int(distance)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance_matrix[j, i] = distance_matrix[i, j]
    # std = np.std(distance_matrix)
    # for i in range(len(distance_matrix)):
    #     distance_matrix[i] = [math.exp(-dist / (2 * std ** 2)) for dist in distance_matrix[i]]

    return distance_matrix


def get_DTW_graph(data):
    nodes_id = [name for name in data.columns.tolist() if name.isdigit()]
    dtw_matrix = np.zeros((len(nodes_id), len(nodes_id)))
    for i in tqdm(range(len(nodes_id))):
        for j in range(i, len(nodes_id)):
            if i == j:
                distance = 0
            else:
                distance, _ = fastdtw(data[nodes_id[i]], data[nodes_id[j]], dist=euclidean)
            dtw_matrix[i, j] = (int(distance))
    for i in range(dtw_matrix.shape[0]):
        for j in range(dtw_matrix.shape[1]):
            dtw_matrix[j, i] = dtw_matrix[i, j]
    # std = np.std(dtw_matrix)
    # for i in range(len(dtw_matrix)):
    #     dtw_matrix[i] = [math.exp(-(dist ** 2) / (2 * std ** 2)) for dist in dtw_matrix[i]]
    return dtw_matrix


def get_QD_trajectory():
    STC = SensorsTrafficConversion()
    bounds = config.QD_Bounds
    params = transbigdata.area_to_params(bounds, accuracy=500, method='rect')
    flow = pd.read_csv('./data/QD_flow_use.csv')
    nodes = [node for node in flow.columns.tolist() if node.isdigit()]
    grid = pd.read_excel('./data/node_location/QD_nodes.xlsx')
    nodes_grid = [grid[grid['crossroadID'].isin([int(node)])]['grid_id'].iloc[0] for node in nodes]

    time_step_geohash = []
    for name in tqdm(os.listdir(config.Train_Taxi_Path), desc='QD word embedding'):
        df_trajectory = pd.read_csv(os.path.join(config.Train_Taxi_Path, name))
        df_count = df_trajectory.groupby(['taxiID']).size().reset_index()
        df_count = df_count.rename(columns={0: 'num'})
        # taxi_id = df_count[(df_count['num'] < 2500) & (df_count['num'] > 500)]['taxiID'].tolist()
        # df_trajectory = df_trajectory[df_trajectory['taxiID'].isin(taxi_id)]
        df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=4, method=STC.multiprocessing_change_coordinate)
        df_trajectory = transbigdata.clean_outofbounds(df_trajectory, bounds=config.QD_Bounds, col=['lng', 'lat'])
        df_trajectory = df_trajectory.sort_values(by=['taxiID', 'timestamp']).reset_index(drop=True)
        df_trajectory = transbigdata.traj_clean_drift(df_trajectory, col=['taxiID', 'timestamp', 'lng', 'lat'], method='twoside',
                                                      speedlimit=80, dislimit=1000, anglelimit=30)
        # df_trajectory = transbigdata.traj_densify(df_trajectory, col=['taxiID', 'timestamp', 'lng', 'lat'], timegap=15)
        df_trajectory = transbigdata.traj_clean_redundant(df_trajectory, col=['taxiID', 'timestamp', 'lng', 'lat'])
        # map_con, edges_p, edges = read_openstreetmap()
        # df_trajectory = hmm_matching(df_trajectory, map_con, edges_p)
        # df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=4, method=STC.multiprocessing_geometry_map_matching)
        df_trajectory['grid_id'] = transbigdata.geohash_encode(df_trajectory['lng'], df_trajectory['lat'], precision=6)

        # ------------ traj sequence -----------------
        df_trajectory["timestamp"] = pd.to_datetime(df_trajectory["timestamp"])
        file_date = datetime.strptime(str(df_trajectory["timestamp"].min())[:11] + "06:59:59", "%Y-%m-%d %H:%M:%S")
        df_trajectory["time_slice"] = [math.ceil((x - file_date).seconds // 300) for x in df_trajectory["timestamp"].tolist()]
        
        for i in range(12, 144 - 12 + 1):
            start_slice = i - 11
            end_slice = i
            
            temp = df_trajectory[df_trajectory['time_slice'].between(start_slice, end_slice)]
            trajectory_res = []
            for taxiID in temp['taxiID'].unique().tolist():
                traj_slice = temp[temp['taxiID'] == taxiID]['grid_id'].tolist()
                traj_slice = [x[0] for x in groupby(traj_slice)]
                if len(traj_slice) > 3:
                    trajectory_res.append(traj_slice)

            time_step_geohash.append(trajectory_res)
    return time_step_geohash


def get_CD_trajectory():
    CDTTC = ChengDuTaxiTrafficConversion()
    bounds = config.CD_Bounds
    params = transbigdata.area_to_params(bounds, accuracy=500, method='rect')
    flow = pd.read_csv('./data/original_flow/CD_flow.csv')
    nodes = [node for node in flow.columns.tolist() if node.isdigit()]
    grid = pd.read_csv('./data/node_location/CD_railway.csv')
    nodes_grid = [grid[grid['node_id'].isin([int(node)])]['grid_id'].iloc[0] for node in nodes]

    time_step_matrix = []
    for name in tqdm(os.listdir(config.CD_Taxi_Path), desc='CD word embedding'):
        df_trajectory = pd.read_pickle(os.path.join(config.CD_Taxi_Path, name))
        df_trajectory = transbigdata.clean_outofbounds(df_trajectory, bounds=config.CD_Bounds, col=['lng', 'lat'])
        df_trajectory = df_trajectory.sort_values(by=['vin', 'time']).reset_index(drop=True)
        df_trajectory = transbigdata.traj_clean_drift(df_trajectory, col=['vin', 'time', 'lng', 'lat'], method='twoside',
                                                      speedlimit=80, dislimit=1000, anglelimit=30)
        df_trajectory = transbigdata.clean_taxi_status(df_trajectory, col=['vin', 'time', 'status'], timelimit=None)
        oddata = transbigdata.taxigps_to_od(df_trajectory, col=['vin', 'time', 'lng', 'lat', 'status'])
        data_deliver, _ = transbigdata.taxigps_traj_point(df_trajectory, oddata, col=['vin', 'time', 'lng', 'lat', 'status'])
        del df_trajectory
        # data_deliver = transbigdata.traj_densify(data_deliver, col=['vin', 'time', 'lng', 'lat'], timegap=15)
        data_deliver = transbigdata.traj_clean_redundant(data_deliver, col=['vin', 'time', 'lng', 'lat'])
        # gcj02 --> wgs08
        data_deliver = run_task_multiprocessing(data=data_deliver, threads=4, method=multiprocessing_change_coordinate)
        data_deliver = data_deliver.drop(columns=['status'])
        data_res = pd.DataFrame()
        slice_num = 4
        for i in tqdm(range(0, slice_num)):
            if i == 0:
                continue
            temp = data_deliver.iloc[i * math.ceil(len(data_deliver) / slice_num): (i + 1) * math.ceil(len(data_deliver) / slice_num)]
            temp = run_task_multiprocessing(data=temp, threads=8, method=CDTTC.multiprocessing_geometry_map_matching)
            data_res = data_res.append(temp).reset_index(drop=True)
        data_deliver = data_res
        del data_res
        # G = ox.load_graphml(r'./data/roadnet/chengdu.graphml')
        # data_deliver = transbigdata.traj_mapmatch(data_deliver, G, col=['lng', 'lat'])
        data_deliver['LONCOL'], data_deliver['LATCOL'] = transbigdata.GPS_to_grid(data_deliver['lng'], data_deliver['lat'], params)
        data_deliver['grid_id'] = data_deliver.apply(lambda x: str(x['LONCOL']) + '_' + str(x['LATCOL']), axis=1)


        data_deliver["time"] = pd.to_datetime(data_deliver["time"])
        file_date = datetime.strptime(str(data_deliver["time"].min())[:11] + "05:59:59", "%Y-%m-%d %H:%M:%S")
        data_deliver["time_slice"] = [math.ceil((x - file_date).seconds // 300) for x in data_deliver["time"].tolist()]
        for i in range(0, 216 - config.CD_config['time_step'] - config.CD_config['pred_step'] + 1):
            temp = data_deliver[data_deliver['time_slice'] == i]
            # temp = temp[['taxiID', 'grid_id']]
            trajectory_res = []
            for traj_id in temp['ID'].unique().tolist():
                traj_slice = temp[temp['ID'] == traj_id]['grid_id'].tolist()
                traj_slice = [x[0] for x in groupby(traj_slice)]
                if len(traj_slice) > 1:
                    trajectory_res.append(traj_slice)
            matrix = transfer_probability(trajectory_res, nodes_grid)
            time_step_matrix.append(matrix)

    return np.array(time_step_matrix)


def traj_word2vec(traj_slice, nodes_encode):
    model = Word2Vec(traj_slice, vector_size=256, window=3, min_count=3, epochs=50, negative=10, sg=1)
    node_embedding = []
    i = 0
    for node in nodes_encode:
        try:
            res = model.wv.get_vector(node)
        except:
            i += 1
            print(f'-- this node {i} is OOV --')
            res = [0] * 256
        node_embedding.append(res)
    return node_embedding


def transfer_probability(traj_slice, nodes_encode):
    model = Word2Vec(traj_slice, vector_size=256, window=6, min_count=1, epochs=50, negative=10, sg=1)
    node_embedding = []
    i = 0
    for node in nodes_encode:
        try:
            res = model.wv.get_vector(node)
        except:
            i += 1
            res = [0] * 256
        node_embedding.append(res)
    print(f'-- have {i} nodes is OOV --')
    # matrix = np.diag([1.0] * len(node_embedding))
    matrix = np.zeros((len(node_embedding), len(node_embedding))).astype(np.float32)
    for i in range(len(node_embedding)-1):
        for j in range(i+1, len(node_embedding)):
            if list(set(node_embedding[i])) == [0] or list(set(node_embedding[j])) == [0]:
                matrix[i, j] = 0
            else:
                # cos_sim = abs(1 - spatial.distance.cosine(node_embedding[i], node_embedding[j]))
                dist = np.linalg.norm(np.array(node_embedding[i]) - np.array(node_embedding[j]))
                matrix[i, j] = dist

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[j, i] = matrix[i, j]

    std, mean = np.std(matrix), np.mean(matrix)
    for i in range(len(matrix)):
        matrix[i] = [math.exp(-(x ** 2) / (2 * std ** 2)) if x != 0 else 0 for x in matrix[i]]
    return matrix + np.diag([1] * len(matrix))