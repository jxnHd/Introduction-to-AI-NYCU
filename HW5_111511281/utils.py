import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def parse_feature_string(feature_str):
    """Parses the string representation of list into a numpy array."""
    # The features are stored as "[-0.1, ...]", json.loads handles this safely and quickly
    return json.loads(feature_str)

def load_data(root_dir='.'):
    print("Loading raw CSV files...")
    # Load raw CSVs
    train_df = pd.read_csv(f'{root_dir}/train.csv')
    test_df = pd.read_csv(f'{root_dir}/test.csv')
    graph_df = pd.read_csv(f'{root_dir}/treads_graph.csv')

    print("Processing features...")
    # Combine node IDs to create a global mapping
    # Note: train and test node_ids should cover all nodes, but let's be safe
    # The problem states 147,328 total users.
    
    # Process Train Features
    train_ids = train_df['node_id'].values
    train_y = train_df['label'].values
    
    # We need to parse all features. 
    # Let's create a dictionary mapping node_id -> feature vector
    # This might be memory intensive, but robust.
    
    # Pre-allocate feature matrix
    # Total unique nodes = union of graph, train, and test
    all_node_ids = set(train_df['node_id']).union(set(test_df['node_id'])).union(set(graph_df['src_node'])).union(set(graph_df['dst_node']))
    sorted_node_ids = sorted(list(all_node_ids))
    
    # Map real node_id to index 0..N-1
    node_map = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
    num_nodes = len(sorted_node_ids)
    feature_dim = 128
    
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    
    # Fill features from train
    print("Parsing train features...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        mapped_idx = node_map[row['node_id']]
        feats = parse_feature_string(row['feature'])
        x[mapped_idx] = torch.tensor(feats, dtype=torch.float)
        
    # Fill features from test
    print("Parsing test features...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        mapped_idx = node_map[row['node_id']]
        feats = parse_feature_string(row['feature'])
        x[mapped_idx] = torch.tensor(feats, dtype=torch.float)

    print("Building Graph Edges...")
    # Process Edges
    # src -> dst
    src = [node_map[i] for i in graph_df['src_node']]
    dst = [node_map[i] for i in graph_df['dst_node']]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Convert to undirected (very important for performance in social graphs)
    edge_index = to_undirected(edge_index)

    print("Creating Masks...")
    # Create Train/Test Masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    y = torch.full((num_nodes,), -1, dtype=torch.long) # Initialize with -1

    # Fill Train Mask and Labels
    for node_id, label in zip(train_ids, train_y):
        mapped_idx = node_map[node_id]
        train_mask[mapped_idx] = True
        y[mapped_idx] = label

    # Fill Test Mask
    test_ids_mapped = []
    for node_id in test_df['node_id']:
        mapped_idx = node_map[node_id]
        test_mask[mapped_idx] = True
        test_ids_mapped.append(node_id) # Keep original IDs for submission

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_classes = 40 # 0 to 39
    
    # Store original IDs for submission mapping
    data.test_node_ids_original = np.array(test_ids_mapped)
    data.id_map_reverse = {v: k for k, v in node_map.items()}

    return data