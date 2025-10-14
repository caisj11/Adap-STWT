# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/29 23:56
# @Author : caisj
# @Email : cai.sj@foxmail.com
# @File : tools.py
# @Software: PyCharm
from math import sin, asin, cos, radians, fabs, sqrt
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from leuvenmapmatching.matcher.distance import DistanceMatcher
import geopandas as gpd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import osmnx as ox
from shapely.geometry import Point, LineString
from leuvenmapmatching.map.inmem import InMemMap


def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1, EARTH_RADIUS = 6371.39):
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance

def get_extreme_value(lng, lat, dist=250):
    lat_max = lat + 180 * dist / (6371.39 * 1000 * np.pi)
    lat_min = lat - 180 * dist / (6371.39 * 1000 * np.pi)
    lng_max = lng + 180 * dist / (6371.39 * 1000 * np.pi * cos(radians(lat)))
    lng_min = lng - 180 * dist / (6371.39 * 1000 * np.pi * cos(radians(lat)))
    return lat_max, lat_min, lng_max, lng_min

def data_smooth(x, step):
    from tsmoothie import KalmanSmoother
    smoother = KalmanSmoother(component='level_longseason', component_noise={'level': 0.3, 'longseason': 0.2}, n_longseasons=step)
    smooth_data = smoother.smooth(x).smooth_data.tolist()[0]
    return smooth_data

def drop_small_flow(df_flow, flow_threshold):
    drop_columns = []
    for column_name in df_flow.columns.tolist():
        if column_name.isdigit():
            if df_flow[column_name].mean() < flow_threshold:
                drop_columns.append(column_name)
    df_flow = df_flow.drop(columns=drop_columns)
    return df_flow


def metric_func(pred, y, times):
    result = {}
    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)

    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:, i, :]
        pred_i = pred[:, i, :]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE
    return result


def get_mae(y_pred, y_true):
    non_zero_pos = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.fabs((y_true[non_zero_pos] - y_pred[non_zero_pos])).mean()


def get_rmse(y_pred, y_true):
    non_zero_pos = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.sqrt(np.square(y_true[non_zero_pos] - y_pred[non_zero_pos]).mean())


def get_mape(y_pred, y_true):
    non_zero_pos = (np.fabs(y_true) > 0.5)
    return np.fabs((y_true[non_zero_pos] - y_pred[non_zero_pos]) / y_true[non_zero_pos]).mean()


def get_r2(y_pred, y_true):
    non_zero_pos = y_true != 0
    y_true_filtered = y_true[non_zero_pos]
    y_pred_filtered = y_pred[non_zero_pos]
    
    if len(y_true_filtered) == 0:
        return 0.0
    
    ss_res = np.sum((y_true_filtered - y_pred_filtered) ** 2)
    ss_tot = np.sum((y_true_filtered - np.mean(y_true_filtered)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def hmm_matching(data, map_con, edges_p, min_coverage_ratio=0.8):
    result_list = []
    for traj_id in data['taxiID'].unique().tolist():
        df_traj = data[data['taxiID'].isin([traj_id])].copy()
        
        if len(df_traj) > 1:
            vin = df_traj['taxiID'].iloc[0]
            original_count = len(df_traj)
            
            df_traj['geometry'] = gpd.points_from_xy(df_traj['lng'], df_traj['lat'])
            df_traj = gpd.GeoDataFrame(df_traj)
            df_traj.crs = 'EPSG:4326'
            df_traj = df_traj.to_crs(2416)
            
            path = list(zip(df_traj.geometry.y, df_traj.geometry.x))
            
            matcher = DistanceMatcher(map_con,
                                    max_dist=500,
                                    max_dist_init=170,
                                    min_prob_norm=0.0001,
                                    non_emitting_length_factor=0.95,
                                    obs_noise=50,
                                    obs_noise_ne=50,
                                    dist_noise=50,
                                    max_lattice_width=20,
                                    non_emitting_states=True)
            
            states, _ = matcher.match(path, unique=False)
            
            if matcher.path_pred_onlynodes:
                pathdf = pd.DataFrame(matcher.path_pred_onlynodes, columns=['u'])
                pathdf['v'] = pathdf['u'].shift(-1)
                pathdf = pathdf[~pathdf['v'].isnull()]
                pathdf['v'] = pathdf['v'].astype(int)
                pathdf['u'] = pathdf['u'].astype(int)
                
                pathgdf = pd.merge(pathdf, edges_p.reset_index(), on=['u', 'v'], how='left')
                pathgdf = gpd.GeoDataFrame(pathgdf)
                
                if not pathgdf.empty and 'geometry' in pathgdf.columns:
                    pathgdf.crs = 'EPSG:2416'
                    pathgdf_4326 = pathgdf.to_crs(4326)
                    
                    all_coords = []
                    for index, row in pathgdf_4326.iterrows():
                        if row['geometry'] is not None:
                            coords = list(row['geometry'].coords)
                            for coord in coords:
                                all_coords.append([coord[0], coord[1], vin, traj_id])
                    
                    if all_coords and len(all_coords) >= original_count * min_coverage_ratio:
                        original_times = df_traj['timestamp'].tolist()
                        
                        if len(all_coords) > 1:
                            time_stamps = pd.date_range(
                                start=original_times[0], 
                                end=original_times[-1],
                                periods=len(all_coords)
                            ).tolist()
                        else:
                            time_stamps = [original_times[0]]
                        
                        for i, coord in enumerate(all_coords):
                            coord.append(time_stamps[i])
                        
                        for coord in all_coords:
                            coord.append(None)
                            coord.append(None)
                        
                        result_list.extend(all_coords)
                    else:
                        for _, row in df_traj.iterrows():
                            result_list.append([row['lng'], row['lat'], vin, traj_id, row['timestamp'], 
                                              row.get('GPSspeed'), row.get('direction')])
                else:
                    for _, row in df_traj.iterrows():
                        result_list.append([row['lng'], row['lat'], vin, traj_id, row['timestamp'], 
                                          row.get('GPSspeed'), row.get('direction')])
            else:
                for _, row in df_traj.iterrows():
                    result_list.append([row['lng'], row['lat'], vin, traj_id, row['timestamp'], 
                                      row.get('GPSspeed'), row.get('direction')])
    
    result_df = pd.DataFrame(result_list, columns=['lng', 'lat', 'taxiID', 'taxiID_copy', 'timestamp', 'GPSspeed', 'direction'])
    result_df = result_df.drop('taxiID_copy', axis=1)
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    result_df = result_df[['taxiID', 'lng', 'lat', 'GPSspeed', 'direction', 'timestamp']]
    return result_df


def read_openstreetmap():
    G = ox.load_graphml(r'./data/roadnet/qingdao.graphml')
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges['lng'], edges['lat'] = edges.centroid.x, edges.centroid.y
    G_p = ox.project_graph(G, to_crs=2416)
    nodes_p, edges_p = ox.graph_to_gdfs(G_p, nodes=True, edges=True)
    map_con = InMemMap(name='myosm', use_latlon=False, use_rtree=True, index_edges=True)
    for node_id, row in nodes_p.iterrows():
        map_con.add_node(node_id, (row['y'], row['x']))
    for node_id_1, node_id_2, _ in G_p.edges:
        map_con.add_edge(node_id_1, node_id_2)
    return map_con, edges_p, edges





if __name__ == "__main__":
    a = get_extreme_value(120.323312031544, 36.0730417809157)
    print(a)