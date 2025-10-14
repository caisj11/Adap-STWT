# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/19 22:39
# @Author : caisj
import logging
import re, datetime, math
import os
from datetime import datetime
import numpy as np
import pandas as pd
from config import Train_Traffic_Path, Box_Width, Crossroad_Location_Path_Name, \
    Train_Taxi_Path, Crossroad_ID, CD_Taxi_Path
import config
from my_logging import logger
from tqdm import tqdm

from utils.multiprocessing import run_task_multiprocessing
from utils.tools import get_extreme_value
# 地图匹配包
import osmnx as ox
import transbigdata
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import geopandas as gpd

class SensorsTrafficConversion():
    def __init__(self):
        self.traffic_path = Train_Traffic_Path
        self.G = ox.load_graphml(r'./data/roadnet/qingdao.graphml')

    def get_file_name(self, path):
        return os.listdir(path)

    def multiprocessing_change_coordinate(self, ns, index):
        data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
        data[['lng', 'lat']] = data.apply(lambda x: transbigdata.gcj02towgs84(x['lng'], x['lat']), axis=1, result_type='expand')
        return data

    def multiprocessing_geometry_map_matching(self, ns, index):
        data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
        data = transbigdata.traj_mapmatch(data, self.G, col=['lng', 'lat'])
        data = data.drop(columns=['dist', 'u', 'v', 'key', 'geometry'])
        return data

    def get_all_traffic_flow(self):
        all_traffic_flow = pd.DataFrame()
        file_names = self.get_file_name(self.traffic_path)
        for file_name in file_names:
            logger.info(f"Compute {file_name} : vehicles/5min")
            all_data = pd.read_csv(os.path.join(self.traffic_path, file_name))
            file_date = datetime.datetime.strptime(all_data["timestamp"].iloc[0][:11] + "06:59:59", "%Y-%m-%d %H:%M:%S")
            all_data["timestamp"] = pd.to_datetime(all_data["timestamp"])
            all_data["time_slice"] = [math.ceil((x - file_date).seconds / 300) for x in all_data["timestamp"].tolist()]
            all_data = all_data.drop_duplicates(subset=["crossroadID", "vehicleID", "time_slice"]).reset_index(drop=True)
            traffic_flow = all_data.groupby(["crossroadID", "time_slice"]).size().reset_index().rename(columns={0: "flow"})
            day_flow = pd.DataFrame({"time_slice": list(range(1, 145)), "month_day": re.findall("(\d{2}-\d{2})", file_name) * 144})
            for crossroad in tqdm(traffic_flow["crossroadID"].unique().tolist()):
                temp_flow = traffic_flow[traffic_flow["crossroadID"] == crossroad]
                if len(temp_flow) > 0:
                    temp_flow = temp_flow[["time_slice", "flow"]].rename(columns={"flow": crossroad})
                    day_flow = pd.merge(day_flow, temp_flow, on=["time_slice"], how="left")
                else:
                    day_flow[crossroad] = 0
            all_traffic_flow = pd.concat([all_traffic_flow, day_flow], ignore_index=True)
            all_traffic_flow = all_traffic_flow.fillna(0)
        return all_traffic_flow


class TaxiTrafficConversion():
    def __init__(self):
        self.box_width = Box_Width
        self.taxi_path = Train_Taxi_Path
        self.crossroad_location = pd.read_excel(
            Crossroad_Location_Path_Name)
        self.crossroad_id = Crossroad_ID

    def get_file_name(self, path):
        return os.listdir(path)

    def get_box_point(self):
        location_map = {}
        for index, row in self.crossroad_location.iterrows():
            lat, lng = row["lat"], row["lng"]
            lat_max, lat_min, lng_max, lng_min = get_extreme_value(lng=lng, lat=lat, dist=Box_Width / 2)
            location_map[row["crossroadID"]] = [lat_max, lat_min, lng_max, lng_min]
        return location_map

    def match_crossroadName(self, x, location_map):
        lng, lat = x["lng"], x["lat"]
        for crossroadID in location_map.keys():
            if (lat <= location_map[crossroadID][0] and lat >= location_map[crossroadID][1] and lng <=
                    location_map[crossroadID][2] and lng >= location_map[crossroadID][3]):
                return str(crossroadID)
        return "null"

    def multiprocessing_match_box(self, ns, index):
        data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
        location_map = self.get_box_point()
        data["crossroadID"] = data.apply(lambda x: self.match_crossroadName(x, location_map), axis=1)
        return data

    def get_all_taxi_flow_speed(self):
        all_taxi_flow = pd.DataFrame()
        all_taxi_speed = pd.DataFrame()
        file_names = self.get_file_name(self.taxi_path)
        for file_name in file_names:
            logger.info(f"Compute {file_name} : vehicles/5min")
            taxi_data = pd.read_csv(os.path.join(self.taxi_path, file_name))
            taxi_data = taxi_data.drop_duplicates(subset=["taxiID", "lng", "lat"]).reset_index(drop=True)
            taxi_data = run_task_multiprocessing(data=taxi_data, threads=6, method=self.multiprocessing_match_box)
            taxi_data = taxi_data[~taxi_data["crossroadID"].isin(["null"])].reset_index(drop=True)
            file_date = datetime.datetime.strptime(taxi_data["timestamp"].iloc[0][:11] + "06:59:59",
                                                   "%Y-%m-%d %H:%M:%S")
            taxi_data["timestamp"] = pd.to_datetime(taxi_data["timestamp"])
            taxi_data["time_slice"] = [math.ceil((x - file_date).seconds / 300) for x in
                                       taxi_data["timestamp"].tolist()]
            taxi_speed = taxi_data.groupby(["crossroadID", "time_slice"])[
                "GPSspeed"].mean().reset_index().copy().rename(columns={"GPSspeed": "speed"})
            taxi_data = taxi_data.drop_duplicates(subset=["taxiID", "crossroadID", "time_slice"]).reset_index(drop=True)
            taxi_flow = taxi_data.groupby(["crossroadID", "time_slice"]).size().reset_index().rename(columns={0: "flow"})
            day_speed = pd.DataFrame({"time_slice": list(range(1, 146)), "month_day": re.findall("(\d{2}-\d{2})", file_name) * 145})
            day_flow = pd.DataFrame({"time_slice": list(range(1, 146)), "month_day": re.findall("(\d{2}-\d{2})", file_name) * 145})
            for crossroad in self.crossroad_id:
                temp_flow = taxi_flow[taxi_flow["crossroadID"] == crossroad]
                if len(temp_flow) > 0:
                    temp_flow = temp_flow[["time_slice", "flow"]].rename(columns={"flow": crossroad})
                    day_flow = pd.merge(day_flow, temp_flow, on=["time_slice"], how="left")
                else:
                    day_flow[crossroad] = 0
                temp_speed = taxi_speed[taxi_speed["crossroadID"] == crossroad]
                if len(temp_speed) > 0:
                    temp_speed = temp_speed[["time_slice", "speed"]].rename(columns={"speed": crossroad})
                    day_speed = pd.merge(day_speed, temp_speed, on=["time_slice"], how="left")
                else:
                    day_speed[crossroad] = 0
            day_speed, day_flow = day_speed.fillna(0), day_flow.fillna(0)
            all_taxi_flow = all_taxi_flow.append(day_flow)
            all_taxi_speed = all_taxi_speed.append(day_speed)
        return all_taxi_flow, all_taxi_speed


def multiprocessing_change_coordinate(ns, index):
    data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
    data[['lng', 'lat']] = data.apply(lambda x: transbigdata.gcj02towgs84(x['lng'], x['lat']), axis=1, result_type='expand')
    return data


class ChengDuTaxiTrafficConversion():
    def __init__(self):
        self.taxi_path = CD_Taxi_Path
        # self.map_con, self.edges_p, self.edges = self.read_openstreetmap()
        self.node_site = pd.read_csv(config.CD_Crossroad_Location_Path_Name)
        self.G = ox.load_graphml(r'./data/roadnet/chengdu.graphml')
        self.flow_threshold = config.CD_Flow_Threshold

    def get_file_name(self, path):
        return os.listdir(path)

    def get_box_point(self):
        location_map = {}
        for index, row in self.node_site.iterrows():
            lat, lng = row["lat"], row["lng"]
            lat_max, lat_min, lng_max, lng_min = get_extreme_value(lng=lng, lat=lat, dist=Box_Width / 2)
            location_map[row["node_id"]] = [lat_max, lat_min, lng_max, lng_min]
        return location_map

    def match_crossroadName(self, x, location_map):
        lng, lat = x["lng"], x["lat"]
        for node_id in location_map.keys():
            if location_map[node_id][0] >= lat >= location_map[node_id][1] and location_map[node_id][2] >= lng >= location_map[node_id][3]:
                return str(node_id)
        return "null"

    def multiprocessing_match_box(self, ns, index):
        data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
        location_map = self.get_box_point()
        data["node_id"] = data.apply(lambda x: self.match_crossroadName(x, location_map), axis=1)
        return data

    def multiprocessing_geometry_map_matching(self, ns, index):
        data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
        data = transbigdata.traj_mapmatch(data, self.G, col=['lng', 'lat'])
        data = data.drop(columns=['dist', 'u', 'v', 'key', 'geometry'])
        return data

    def multiprocessing_time_slice(self, ns, index):
        data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
        date = data['time'][0].strftime('%Y-%m-%d')
        min_time = datetime.strptime(f'{date} 06:00:00', "%Y-%m-%d %H:%M:%S")
        data["time_slice"] = data.apply(lambda x: math.ceil(((x['time'] - min_time).seconds + 1) / 300), axis=1)
        data["time_slice"] = data["time_slice"].astype('int16')
        return data

    def read_openstreetmap(self):
        G = ox.load_graphml(r'./data/roadnet/chengdu.graphml')
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        edges['lng'], edges['lat'] = edges.centroid.x, edges.centroid.y
        G_p = ox.project_graph(G, to_crs=2416)
        nodes_p, edges_p = ox.graph_to_gdfs(G_p, nodes=True, edges=True)
        map_con = InMemMap(name='myosm', use_latlon=False)  # , use_rtree=True, index_edges=True)
        for node_id, row in nodes_p.iterrows():
            map_con.add_node(node_id, (row['y'], row['x']))
        for node_id_1, node_id_2, _ in G_p.edges:
            map_con.add_edge(node_id_1, node_id_2)
        return map_con, edges_p, edges

    def leuvenmap_map_matching(self, data, map_con, edges_p):
        result = []
        for traj_id in tqdm(data['ID'].unique().tolist()):
            df_traj = data[data['ID'].isin([traj_id])].copy()
            if len(df_traj) > 1:
                vin = df_traj['vin'].iloc[0]
                df_traj = transbigdata.traj_densify(df_traj, col=['vin', 'time', 'lng', 'lat'], timegap=15)
                df_traj['geometry'] = gpd.points_from_xy(df_traj['lng'], df_traj['lat'])
                df_traj = gpd.GeoDataFrame(df_traj)
                df_traj.crs = {'init': 'epsg:4326'}
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
                pathdf = pd.DataFrame(matcher.path_pred_onlynodes, columns=['u'])
                pathdf['v'] = pathdf['u'].shift(-1)
                pathdf = pathdf[-pathdf['v'].isnull()]
                pathgdf = pd.merge(pathdf, edges_p.reset_index())
                pathgdf = gpd.GeoDataFrame(pathgdf)
                # pathgdf.plot()
                pathgdf.crs = {'init': 'epsg:2416'}
                pathgdf_4326 = pathgdf.to_crs(4326)
                vin_res = []
                for index, row in pathgdf_4326.iterrows():
                    temp = [[i[0], i[1], vin, traj_id] for i in row['geometry'].coords[:]]
                    time = pd.date_range(str(df_traj['time'].min()), str(df_traj['time'].max()),
                                         len(temp)).tolist()
                    [temp[i].append(time[i]) for i in range(len(temp))]
                    vin_res.append(temp)

                result.append(vin_res)
        result = pd.DataFrame(result, columns=['lng', 'lat', 'vin', 'ID', 'time'])

        return result

    def trajectory_process(self, data):
        # gcj02 --> wgs08
        data = run_task_multiprocessing(data=data, threads=4, method=multiprocessing_change_coordinate)
        data = transbigdata.clean_outofbounds(data, bounds=config.CD_Bounds, col=['lng', 'lat'])
        data = data.sort_values(by=['vin', 'time']).reset_index(drop=True)
        data = transbigdata.traj_clean_drift(data, col=['vin', 'time', 'lng', 'lat'], method='twoside', speedlimit=80, dislimit=1000, anglelimit=30)
        data = transbigdata.clean_taxi_status(data, col=['vin', 'time', 'status'], timelimit=None)
        data = transbigdata.traj_densify(data, col=['vin', 'time', 'lng', 'lat'], timegap=15)
        data = transbigdata.traj_clean_redundant(data, col=['vin', 'time', 'lng', 'lat'])
        # data = data.drop(columns=['vin', 'status'])
        # logging.info(data.info())
        # data_res = pd.DataFrame()
        # slice_num = 4
        # for i in tqdm(range(0, slice_num)):
        #     if i == 0:
        #         continue
        #     temp = data.iloc[i * math.ceil(len(data) / slice_num): (i + 1) * math.ceil(len(data) / slice_num)]
        #     temp = run_task_multiprocessing(data=temp, threads=8, method=self.multiprocessing_geometry_map_matching)
        #     data_res = data_res.append(temp).reset_index(drop=True)
        return data

    def count_traffic_flow(self, data, begin_time):
        location_map = self.get_box_point()
        data["node_id"] = data.apply(lambda x: self.match_crossroadName(x, location_map), axis=1)
        # data = run_task_multiprocessing(data=data, threads=8, method=self.multiprocessing_match_box)
        data = data[~data['node_id'].isin(['null'])].reset_index(drop=True)
        data["time_slice"] = data['time'].apply(lambda x: math.ceil(((x - begin_time).seconds + 1) / 300))
        data["time_slice"] = data["time_slice"].astype('int16')
        data = data.drop_duplicates(['vin', 'node_id', "time_slice"]).reset_index(drop=True)
        taxi_flow = data.groupby(["node_id", "time_slice"]).size().reset_index().rename(columns={0: "flow"})
        taxi_flow['node_id'] = taxi_flow['node_id'].astype('int')
        day_flow = pd.DataFrame({"time_slice": list(range(1, 217)), "date": [data['time'].iloc[0].strftime('%d')] * 216})
        del data
        for node in tqdm(self.node_site['node_id'].unique().tolist(), desc='grid flow'):
            temp_flow = taxi_flow[taxi_flow["node_id"] == node]
            if len(temp_flow) > 0:
                temp_flow = temp_flow[["time_slice", "flow"]].rename(columns={"flow": node})
                day_flow = pd.merge(day_flow, temp_flow, on=["time_slice"], how="left")
            else:
                day_flow[node] = 0
        day_flow = day_flow.fillna(0)
        return day_flow

    def drop_small_flow(self, df_flow):
        drop_columns = []
        for column_name in df_flow.columns.tolist():
            if str(column_name).isdigit():
                if df_flow[column_name].mean() < self.flow_threshold:
                    drop_columns.append(column_name)
        df_flow = df_flow.drop(columns=drop_columns)
        return df_flow

    def get_all_traffic_flow(self):
        all_taxi_flow = pd.DataFrame()
        file_names = self.get_file_name(self.taxi_path)
        for file_name in file_names:
            logger.info(f"---   {file_name}   ---")
            data = pd.read_pickle(os.path.join(self.taxi_path, file_name))
            begin_time = data['time'].min()
            data = self.trajectory_process(data)
            day_flow = self.count_traffic_flow(data, begin_time)
            day_flow.to_csv(rf"./data/{file_name.split('_')[0]}_flow.csv", index=False)
            all_taxi_flow = all_taxi_flow.append(day_flow)
        all_taxi_flow.to_csv(r'data/CD_taxi_flow.csv', index=False)
        all_taxi_flow = self.drop_small_flow(all_taxi_flow)

        return all_taxi_flow

    def plot_match_result(self, pathgdf_4326, edges):
        import matplotlib.pyplot as plt
        fig = plt.figure(1, (8, 8), dpi=100)
        ax = plt.subplot(111)
        plt.sca(ax)
        fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))
        bounds = pathgdf_4326.unary_union.bounds
        gap = 0.003
        bounds = [bounds[0] - gap, bounds[1] - gap, bounds[2] + gap, bounds[3] + gap]
        pathgdf_4326.plot(ax=ax, zorder=1)
        transbigdata.clean_outofbounds(edges, bounds, col=['lng', 'lat']).plot(ax=ax, color='#333', lw=0.1)
        transbigdata.to_crs(4326).plot(ax=ax, color='r', markersize=5, zorder=2)

        plt.axis('off')
        plt.xlim(bounds[0], bounds[2])
        plt.ylim(bounds[1], bounds[3])
        plt.show()
