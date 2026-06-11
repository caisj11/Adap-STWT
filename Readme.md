<!--
 * @Author: Caisj
 * @Date: 2025-10-13 18:30:10
 * @LastEditTime: 2025-10-14 09:29:37
-->
# Adap-STWT

## 📊 Datasets

### Qingdao Dataset
- **Download Link**: https://aistudio.baidu.com/datasetdetail/27526
- **Description**: Contains traffic data and taxi trajectory data from Qingdao city
- **Data Type**: Traffic flow data and GPS trajectory data

### Chengdu Dataset
- **Download Link**: https://www.pkbigdata.com/common/zhzgbCmptDataDetails.html
- **Description**: Contains taxi trajectory data from Chengdu city
- **Data Type**: GPS trajectory data

## Installation

### Install Dependencies

```bash
pip install -r requirement.txt
```

## 📦 Usage

### Quick Start (Using Preprocessed Data)

If you have downloaded the preprocessed data:

1. Extract the downloaded data to the `data/` directory
2. Run the main training script:

```bash
python run_main.py
```

### Full Pipeline (From Raw Data)

#### Step 1: Data Preprocessing

Process raw traffic and trajectory data for both Qingdao and Chengdu datasets:

```bash
python run_preprocessing.py
```

This script will:
- Load raw traffic flow and trajectory data
- Filter and clean the data
- Generate flow matrices and node information
- Save processed data to `data/flow/` directory

#### Step 2: Learn Microscopic Graphs

Extract graph structures from microscopic trajectory data:

```bash
python run_micro_graph.py
```

This script will:
- Analyze trajectory patterns and compute micro matrices
- Save micro matrices to `data/graph/` directory

#### Step 3: Train and Evaluate the Model

Train the Adap-STWT model and perform prediction:

```bash
python run_main.py
```

This script will:
- Load preprocessed data and graph structures
- Initialize the Adap-STWT model
- Train the model with adaptive graph learning
- Evaluate on test set and report metrics (MAE, RMSE, MAPE)

## Citation
@ARTICLE{11554185,
  author={Cai, Shijie and Hu, Jie and Chen, Jia and Xu, Wencai},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Adap-STWT: Real-Time Traffic Flow Prediction Based on Multi-Scale Graph Adaptive Fusion and Spatio-Temporal Wavelet Transformer}, 
  year={2026},
  pages={1-14},
  doi={10.1109/TITS.2026.3698123},
  ISSN={1558-0016},
  month={},}



