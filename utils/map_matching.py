import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from scipy.stats import norm
import networkx as nx
from shapely.geometry import Point

class HMMMapMatcher:
    def __init__(self, road_network, edges_gdf, sigma=50, beta=0.1):
        self.G = road_network
        self.edges_gdf = edges_gdf.copy()
        self.sigma = sigma
        self.beta = beta
        
        if self.edges_gdf.crs is None or self.edges_gdf.crs.to_string() == 'EPSG:4326':
            self.edges_gdf_proj = self.edges_gdf.to_crs('EPSG:3857')  # Web Mercator for distance
        else:
            self.edges_gdf_proj = self.edges_gdf
        
        self._prepare_edge_points()
    
    def _prepare_edge_points(self):
        self.edge_points = []
        
        for idx, edge in self.edges_gdf.iterrows():
            line_geom = edge.geometry
            line_proj = self.edges_gdf_proj.iloc[idx].geometry

            length = line_proj.length
            if length == 0:
                continue

            num_points = max(2, int(length / 50) + 1)
            
            for i in range(num_points):
                ratio = i / (num_points - 1) if num_points > 1 else 0
                
                interpolated_point = line_geom.interpolate(ratio, normalized=True)
                
                self.edge_points.append({
                    'point_wgs84': interpolated_point,
                    'edge_u': edge['u'],
                    'edge_v': edge['v'],
                    'position_ratio': ratio,
                    'edge_idx': idx
                })
    
    def _get_candidates_for_gps_point(self, gps_point, max_dist=300):
        gps_geom_wgs84 = Point(gps_point[1], gps_point[0])  # lng, lat
        gps_geom_proj = gpd.GeoSeries([gps_geom_wgs84], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
        
        candidates = []
        
        for edge_point in self.edge_points:
            point_proj = gpd.GeoSeries([edge_point['point_wgs84']], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
            distance = gps_geom_proj.distance(point_proj)
            
            if distance <= max_dist:
                candidates.append({
                    'point_wgs84': edge_point['point_wgs84'],
                    'edge_u': edge_point['edge_u'],
                    'edge_v': edge_point['edge_v'],
                    'position_ratio': edge_point['position_ratio'],
                    'edge_idx': edge_point['edge_idx'],
                    'distance_to_gps': distance,
                    'lat': edge_point['point_wgs84'].y,
                    'lng': edge_point['point_wgs84'].x
                })

        candidates.sort(key=lambda x: x['distance_to_gps'])
        return candidates[:30]
    
    def _emission_probability(self, distance):
        if distance == 0:
            return 1.0
        return np.exp(-distance / self.sigma)
    
    def _get_road_distance(self, cand1, cand2):
        if cand1['edge_idx'] == cand2['edge_idx']:
            edge_geom_proj = self.edges_gdf_proj.iloc[cand1['edge_idx']].geometry
            edge_length = edge_geom_proj.length
            position_diff = abs(cand2['position_ratio'] - cand1['position_ratio'])
            return position_diff * edge_length
        else:
            point1_proj = gpd.GeoSeries([cand1['point_wgs84']], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
            point2_proj = gpd.GeoSeries([cand2['point_wgs84']], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
            euclidean_dist = point1_proj.distance(point2_proj)
            return euclidean_dist * 1.4
    
    def _transition_probability(self, prev_candidate, curr_candidate, gps_distance):
        if gps_distance == 0:
            return 0.5
        road_distance = self._get_road_distance(prev_candidate, curr_candidate)
        distance_diff = abs(road_distance - gps_distance)
        normalized_diff = distance_diff / gps_distance
        
        return np.exp(-self.beta * normalized_diff)
    
    def _viterbi_algorithm(self, gps_points):
        n_obs = len(gps_points)
        if n_obs == 0:
            return []

        all_candidates = []
        for gps_point in gps_points:
            candidates = self._get_candidates_for_gps_point(gps_point)
            if not candidates:
                candidates = [{
                    'point_wgs84': Point(gps_point[1], gps_point[0]),
                    'lat': gps_point[0],
                    'lng': gps_point[1],
                    'distance_to_gps': 0,
                    'is_original': True
                }]
            all_candidates.append(candidates)

        if n_obs == 1:
            best_candidate = all_candidates[0][0]
            return [(best_candidate['lat'], best_candidate['lng'])]

        V = []
        path = []

        V.append({})
        path.append({})
        for i, candidate in enumerate(all_candidates[0]):
            emission_prob = self._emission_probability(candidate['distance_to_gps'])
            V[0][i] = np.log(emission_prob + 1e-10)
            path[0][i] = None

        for t in range(1, n_obs):
            V.append({})
            path.append({})

            prev_gps = gps_points[t-1]
            curr_gps = gps_points[t]
            prev_point_proj = gpd.GeoSeries([Point(prev_gps[1], prev_gps[0])], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
            curr_point_proj = gpd.GeoSeries([Point(curr_gps[1], curr_gps[0])], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
            gps_distance = prev_point_proj.distance(curr_point_proj)
            
            for curr_i, curr_candidate in enumerate(all_candidates[t]):
                emission_prob = self._emission_probability(curr_candidate['distance_to_gps'])
                log_emission = np.log(emission_prob + 1e-10)
                
                max_log_prob = float('-inf')
                best_prev_state = None
                
                for prev_i in V[t-1]:
                    prev_candidate = all_candidates[t-1][prev_i]

                    if 'is_original' in prev_candidate or 'is_original' in curr_candidate:
                        trans_prob = 0.1
                    else:
                        trans_prob = self._transition_probability(prev_candidate, curr_candidate, gps_distance)
                    
                    log_trans = np.log(trans_prob + 1e-10)
                    total_log_prob = V[t-1][prev_i] + log_trans + log_emission
                    
                    if total_log_prob > max_log_prob:
                        max_log_prob = total_log_prob
                        best_prev_state = prev_i
                
                if best_prev_state is not None:
                    V[t][curr_i] = max_log_prob
                    path[t][curr_i] = best_prev_state
        
        if not V[n_obs-1]:
            return gps_points

        max_log_prob = float('-inf')
        best_last_state = None
        for state_i in V[n_obs-1]:
            if V[n_obs-1][state_i] > max_log_prob:
                max_log_prob = V[n_obs-1][state_i]
                best_last_state = state_i
        
        if best_last_state is None:
            return gps_points
        
        optimal_states = []
        state = best_last_state
        
        for t in range(n_obs-1, -1, -1):
            optimal_states.append(state)
            if t > 0 and state in path[t]:
                state = path[t][state]
            else:
                state = None
        
        optimal_states.reverse()

        result = []
        for t, state_i in enumerate(optimal_states):
            if state_i is not None and state_i < len(all_candidates[t]):
                candidate = all_candidates[t][state_i]
                result.append((candidate['lat'], candidate['lng']))
            else:
                result.append(gps_points[t])
        
        return result
    
    def match_trajectory(self, gps_points):
        if len(gps_points) < 1:
            return []
        
        if len(gps_points) == 1:
            candidates = self._get_candidates_for_gps_point(gps_points[0])
            if candidates:
                best_candidate = candidates[0]
                return [(best_candidate['lat'], best_candidate['lng'])]
            else:
                return gps_points

        matched_points = self._viterbi_algorithm(gps_points)

        if len(matched_points) != len(gps_points):
            return gps_points
        
        return matched_points

def hmm_map_matching(data, road_network, edges_gdf):
    result_list = []

    matcher = HMMMapMatcher(road_network, edges_gdf, sigma=80, beta=0.05)
    
    for taxi_id in tqdm(data['taxiID'].unique().tolist(), desc="HMM Map Matching"):
        df_traj = data[data['taxiID'] == taxi_id].copy()
        
        if len(df_traj) > 0:
            df_traj = df_traj.sort_values('timestamp').reset_index(drop=True)
            gps_points = [(row['lat'], row['lng']) for _, row in df_traj.iterrows()]
            matched_points = matcher.match_trajectory(gps_points)
            for i in range(len(df_traj)):
                row = df_traj.iloc[i]
                
                if i < len(matched_points):
                    matched_lat, matched_lng = matched_points[i]
                else:
                    matched_lat, matched_lng = row['lat'], row['lng']
                
                result_list.append([
                    matched_lng,  # lng
                    matched_lat,  # lat
                    row['taxiID'],
                    row['taxiID'], 
                    row['timestamp'],
                    row.get('GPSspeed'),
                    row.get('direction')
                ])

    if not result_list:
        return pd.DataFrame(columns=['taxiID', 'lng', 'lat', 'GPSspeed', 'direction', 'timestamp'])
    
    result_df = pd.DataFrame(result_list, columns=['lng', 'lat', 'taxiID', 'taxiID_copy', 'timestamp', 'GPSspeed', 'direction'])
    result_df = result_df.drop('taxiID_copy', axis=1)
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    result_df = result_df[['taxiID', 'lng', 'lat', 'GPSspeed', 'direction', 'timestamp']]
    
    return result_df

def multiprocessing_custom_hmm_matching(self, ns, index):
    data = ns.data.iloc[index[0]:] if len(index) == 1 else ns.data.iloc[index[0]:index[1]].copy()
    matched_data = hmm_map_matching(data, self.G, self.edges_gdf)
    return matched_data