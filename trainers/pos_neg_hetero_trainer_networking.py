"""Implement trainer that oversees end-to-end training process"""

import os
import logging
from copy import deepcopy
import time

from typing import Any, Dict, List, Tuple
import torch
from tqdm import tqdm
import torch.distributed as dist


from models.newsSAGE import NewsSAGEModel
# from models.utils.layers import seed_everything
from performance import PerformanceStore
# from trainers.metrics import auc, mrr, nDCG
# import torch.nn.functional as F


from comm_utils import (
    sync_model,
    MultiThreadReducerCentralized,
)


logger = logging.getLogger(__name__)
comm_volume_perf_store = PerformanceStore()


def set_torch_seed(seed):
    """Set the seed for torch"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model(entire_model, curr_layer, device):
    """Set up the model"""
    # train the nth layer of the model
    curr_model = entire_model.get_nth_layer(curr_layer)
    if device == "cuda":
        curr_model = curr_model.cuda()
    return curr_model



def init_master(cfg, hydra_output_dir):
    """"Initialize the master process"""
    
    addr = cfg.distributed.master_addr
    port = cfg.distributed.master_port
    
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend='gloo', 
                            init_method=f'tcp://{addr}:{port}',
                            rank=0, world_size=cfg.num_partitions+1)
    print("Distributed setup initialized")
    # seed_everything(cfg.seed)
    
    node_emb_meta = {
        'user': {
            'Category': 384,
            'SubCategory': 384,
            # 'Node2Vec': 128,
        },
        'news': {
            'News_Title_Embedding': 384,
            'News_Abstract_Embedding': 384,
            'Category': 384,
            'SubCategory': 384,
            # 'Node2Vec': 128,
        },
        'entity': {
            'Entity_Embedding': 100,
            # 'Node2Vec': 128,
        },
    }
    
    global_model = NewsSAGEModel(
        cfg.adaptor_hidden, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, node_emb_meta, "cpu", cfg.num_layers, cfg.cross_score, cfg.dropout
    )
    
    perf_stores = []
    
    # oversee gradient aggregation
    for layer in range(len(cfg.num_rounds)):
        model = setup_model(global_model, layer, "cpu")
        
        sync_model(model)
        reducer = MultiThreadReducerCentralized(model=model, device=cfg.device) # TODO: add parameters and model
        
        print("Starting layer {}...".format(layer))
        if cfg.enclave:
            reducer.master_aggregate_enclave(cfg, layer, perf_stores)
        else:
            # reducer.master_aggregate_gradients(cfg, layer, perf_stores)
            reducer.master_aggregate_plain(cfg, layer, perf_stores)
        print("Gradients update finished.")
        
        for i, perf_store in enumerate(perf_stores):
            print("Layer {}:".format(i))
            print("Mean grad reduce time: {} STD: {}".format(perf_store.get_mean_grad_reduce_times(), perf_store.get_std_grad_reduce_times()))
            print("Mean grad decryption time: {}".format(perf_store.get_mean_grad_decryption()))

    
        # print("Aggregating news embeddings from users...")
        # reducer.master_aggregate_users(cfg)
        
        
        print("News embeddings updated and sent.")
        print("Layer {} finished.".format(layer))
        

    for i, perf_store in enumerate(perf_stores):
        print("Layer {}:".format(i))
        print("Mean grad reduce time: {}".format(perf_store.get_mean_grad_reduce_times()))
        print("Mean grad decryption time: {}".format(perf_store.get_mean_grad_decryption()))

