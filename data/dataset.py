"""Implement dataset related functions"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


from data.mind_dataset import MIND_DGL
import dgl # type: ignore

logger = logging.getLogger(__name__)



def load_data(dataset_name: str, dataset_dir: str, cfg=None, add_self_loop=False) -> Tuple[dgl.DGLGraph, int, int]:
    """Load dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name
    dataset_dir : str
        Directory to save the dataset
    add_self_loop : bool, optional

    Returns
    -------
    dgl.DGLGraph
        Graph in DGL format
    """
    graph = dgl.DGLGraph()
    if dataset_name == "mind":
        dataset = MIND_DGL(cfg, force_reload=False)
        graph = dataset.graph
        return graph, dataset
        # graph.ndata["label"] = graph.ndata["label"].float()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

def _partition_graph(graph, shuffled_users, num_partitions):
    num_users = graph.num_nodes("user")

    # Split users into partitions
    users_per_partition = num_users // num_partitions
    user_partitions = np.array_split(shuffled_users, num_partitions)
    
    partitioned_graphs = []
    id_maps = []
    edge_id_maps = []
    
    for partition_users in user_partitions:
        user_mask = torch.zeros(graph.num_nodes('user'), dtype=torch.bool)
        user_mask[partition_users] = True
        all_edges = []
        
        # Iterate over all edge types
        for etype in graph.etypes:
            src_type, _, dst_type = graph.to_canonical_etype(etype)
            
            # If users are the source, get out edges
            if src_type == 'user':
                edges = graph.out_edges(partition_users, etype=etype, form='eid')
                all_edges.append((etype, edges))
            
            # If users are the destination, get in edges
            if dst_type == 'user':
                edges = graph.in_edges(partition_users, etype=etype, form='eid')
                all_edges.append((etype, edges))
        
        # Create the partitioned graph including users and their neighbors
        partitioned_graph = graph.edge_subgraph(dict(all_edges))

        # Create ID mappings
        id_map = {}
        for ntype in partitioned_graph.ntypes:
            orig_ids = partitioned_graph.nodes[ntype].data[dgl.NID]
            new_ids = torch.arange(partitioned_graph.num_nodes(ntype))
            id_map[ntype] = dict(zip(new_ids.tolist(), orig_ids.tolist()))
        
        # Create edge ID mappings
        edge_id_map = {}
        for etype in partitioned_graph.etypes:
            local_edge_ids = torch.arange(partitioned_graph.num_edges(etype))
            global_edge_ids = partitioned_graph.edges[etype].data[dgl.EID]
            edge_id_map[etype] = dict(zip(local_edge_ids.tolist(), global_edge_ids.tolist()))
        
        
        partitioned_graphs.append(partitioned_graph)
        id_maps.append(id_map)
        edge_id_maps.append(edge_id_map)
    
    return partitioned_graphs, id_maps, edge_id_maps


def create_partitions(cfg): 
    graph, dataset = load_data(**cfg.dataset.download, cfg=cfg)
    num_partitions = cfg.num_partitions
    
    user_nodes = graph.nodes("user")
    
    # Randomly shuffle user nodes
    shuffled_users = np.random.permutation(user_nodes.numpy())
    graph_partitions, id_maps, edge_id_maps = _partition_graph(graph, shuffled_users, num_partitions)
    
    items_per_partition = []
    
    for i in range(2, cfg.num_partitions):
        sub_graph = graph_partitions[i]
        items_per_partition.append(sub_graph.num_nodes("news"))
        
        print(f"Partition {i}: {sub_graph.num_nodes('user')} users, {sub_graph.num_nodes('news')} news, {sub_graph.num_edges()} edges")
        id_map = id_maps[i]
        edge_map = edge_id_maps[i]
        
        s3_bucket = cfg.s3_bucket if cfg.s3_bucket != "" else '.'
        s3_path = f"{s3_bucket}/partitions/{cfg.num_partitions}/{i}"
        # s3_path = f"datasets/partitions/{i}"
        dataset.save_partition(sub_graph, id_map, edge_map, s3_path)
        

        
        
        
        # Note: if I map the edge IDs in the sessions (and I assume the loaders too) using the global to local edge ID map I can get the right edges in the new subgraph to do the evaluation. (true story)
        
    print("MEAN = ", np.mean(items_per_partition))
    print("STD = ", np.std(items_per_partition))