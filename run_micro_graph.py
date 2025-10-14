

import json
import numpy as np
from tqdm import tqdm
from model.AdapSTWT.graph_learn import transfer_probability

# read traj data (geohash) and transfer to probability matrix
with open(r'data/qd_node_geohash.json', "r", encoding="utf-8") as f:
    nodes_grid = json.load(f)

data_path = r'./data/flow/QD'
names = ('train_traj.json', 'val_traj.json', 'test_traj.json')
for name in names:
    traj_path = data_path + '/' + name
    with open(traj_path, "r", encoding="utf-8") as f:
        traj_data = json.load(f)
    traj_mx = []
    for i in tqdm(range(len(traj_data))):
        matric = transfer_probability(traj_mx=traj_data[i], nodes_grid=nodes_grid, vector_size=100,
                                            window=5, min_count=1, negative=5, epochs=20, lr=0.01)
        traj_mx.append(matric)
    traj_mx = np.array(traj_mx)
    print(traj_mx.shape)
    np.save('./data/flow/QD/' + name.split('.')[0] + '.npy', traj_mx)
