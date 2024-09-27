"""Implement trainer that oversees end-to-end training process"""

import wandb
import os
import logging
from copy import deepcopy
import time

from typing import Any, Dict, List, Tuple
import dgl # type: ignore
import torch
from tqdm import tqdm
import torch.distributed as dist

from omegaconf import DictConfig
from data.dataset import load_data

from models.newsSAGE import NewsSAGEModel
from models.utils.layers import seed_everything
from performance import PerformanceStore

from trainers.metrics import auc, mrr, nDCG
import torch.nn.functional as F

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


from comm_utils import (
    get_boundary_nodes,
    send_and_receive_embeddings,
    aggregate_metrics,
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


def train(
    graph: dgl.DGLGraph,
    dataset,
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
) -> None:
    """Implement end-to-end training process"""
    # set the seed
    set_torch_seed(cfg.seed)
    base_etypes = ['history', 'history_r', 'ne_link', 'ne_link_r', 'ue_link', 'ue_link_r']
    node_emb_meta = {
        'user': {
            'Category': 384,
            'SubCategory': 384,
        },
        'news': {
            'News_Title_Embedding': 384,
            'News_Abstract_Embedding': 384,
            'Category': 384,
            'SubCategory': 384,
        },
        'entity': {
            'Entity_Embedding': 100,
        },
    }
    
    wandb.init(project=f"{cfg.app}-{cfg.dataset_name}", config=dict(cfg), mode=cfg.wandb_mode)
    
    for ntype in dataset.num_node:
        dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([dataset.num_node[ntype], cfg['hidden_dim'] * 2]).float()
    for etype in dataset.num_relation:
        dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones([dataset.num_relation[etype]]).float() * 0.5

    base_canonical_etypes = sorted([canonical_etype for canonical_etype in dataset.graph.canonical_etypes if canonical_etype[1] in base_etypes])


    # set up the model
    device = cfg.device
    perf_stores = [PerformanceStore()]
    perf_store = perf_stores[0]

    # rank, _ = dist.get_rank(), dist.get_world_size()

    log_dir = os.path.join(hydra_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)


    # Model
    num_layers = cfg.num_layers
    model = NewsSAGEModel(
        cfg.adaptor_hidden, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, node_emb_meta, device, num_layers, cfg.cross_score, cfg.dropout
    ).to(device)


    if not cfg.retexo:
        perf_metrics, epoch, best_epoch = train_end2end(model, dataset, cfg, device, log_dir, hydra_output_dir, base_etypes)
    else:
        perf_metrics, epoch, best_epoch = train_retexo(model, dataset, cfg, device, log_dir, hydra_output_dir, base_etypes)
    fstr = '\nDONE after {} iterations\nBest AUC: {} at epoch {}. All metrics: {}'.format(epoch, perf_metrics[0], best_epoch, perf_metrics)
    print(fstr)
    with open(log_dir + "/accuracy.txt", "a+") as f:
        f.write(fstr)
        
def train_retexo(model, dataset, cfg, device, log_dir, hydra_output_dir, base_etypes):
    curr_layer = 0
    acc_training_time = 0
    not_improved = 0
    global_model = model
    
    perf_stores = [PerformanceStore()]
    
    model = setup_model(global_model, curr_layer, cfg.device)
    sync_model(model)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    
    load_time_s = time.time()
    
    # all train blocks (for layer-wise forward passes)
    full_pos_dataloader, full_neg_dataloader = dataset.get_gnn_train_loader(base_etypes, 2)
    assert(len(full_pos_dataloader) == len(full_neg_dataloader) == 1)
    _, final_pos_sample_graph, all_pos_blocks = next(iter(full_pos_dataloader))
    _, final_neg_sample_graph, all_neg_blocks = next(iter(full_neg_dataloader))
    
    # create the full batch of training edges
    pos_dataloader, neg_dataloader = dataset.get_gnn_train_loader(base_etypes, 1)
    assert(len(pos_dataloader) == len(neg_dataloader) == 1)
    _, pos_sample_graph, pos_blocks = next(iter(pos_dataloader))
    _, neg_sample_graph, neg_blocks = next(iter(neg_dataloader))
    
    pos_sample_graph = pos_sample_graph.to(device)
    pos_blocks = [b.to(device) for b in pos_blocks]
    neg_sample_graph = neg_sample_graph.to(device)
    neg_blocks = [b.to(device) for b in neg_blocks]
    
    del pos_dataloader
    del neg_dataloader
    del full_pos_dataloader
    del full_neg_dataloader

    # validation blocks
    user_dataloader, news_dataloader = dataset.get_gnn_dev_node_loader(base_etypes, cfg.num_layers)
    (_, _, user_blocks) = next(iter(user_dataloader))
    val_user_blocks = [b.to(device) for b in user_blocks]
    (_, _, news_blocks) = next(iter(news_dataloader))
    val_news_blocks = [b.to(device) for b in news_blocks]
    
    del user_dataloader
    del news_dataloader
    
    load_time = time.time() - load_time_s
    print(f"Loaded all data in: {load_time}s")

    # Embedding layer (0)
    best_epoch, best_model = train_embedding_layer(model, dataset, cfg, device, log_dir, opt, pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks, val_user_blocks, val_news_blocks, perf_stores)
             
    print("Layer 0 training finished, encoding features for next layer...")
      
    if cfg.best_model:
        model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        # TODO Send the embeddings and item request to the master (use dst_nodes())
        # TODO make a single *item request* to the master for all blocks
        # TODO receive items and store "per block"
        
        
        # # encoding the eval features 
        eval_user_features = [model.encode(val_user_blocks, encode_source=True)]
        eval_news_features = [model.encode(val_news_blocks, encode_source=True)]
        
        middle_user_features = model.encode(val_user_blocks[1:], encode_source=True)
        middle_news_features = model.encode(val_news_blocks[1:], encode_source=True)

        # encoding the training features for last layer
        final_pos_sample_graph = final_pos_sample_graph.to(device)
        all_pos_blocks = [b.to(device) for b in all_pos_blocks]
        final_pos_features = [model.encode(all_pos_blocks,encode_source=True)]
        final_neg_sample_graph = final_neg_sample_graph.to(device)
        all_neg_blocks = [b.to(device) for b in all_neg_blocks]
        final_neg_features = [model.encode(all_neg_blocks,encode_source=True)]
        
        
        # encoding training features for next layer
        pos_features = model.encode(pos_blocks, encode_source=True)
        neg_features = model.encode(neg_blocks, encode_source=True)
    
    torch.save(model, '{}/{}_layer{}_seed={}_ckt={}.pth'.format(
            hydra_output_dir, 
            "mind", 
            curr_layer, 
            cfg.seed,
            best_epoch if cfg.best_model else cfg.num_rounds[curr_layer]
        ))
    
    # Layer 1
    curr_layer += 1
    model = setup_model(global_model, curr_layer, cfg.device)
    sync_model(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[curr_layer])
        
        
    best_epoch, best_model = train_gnn_layer(model, dataset, cfg, device, log_dir, curr_layer, opt, pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks, val_user_blocks, val_news_blocks, middle_user_features, middle_news_features, pos_features, neg_features, perf_stores)
        
        
    if cfg.best_model:
        model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        # message passing eval features for last layer
        eval_user_features.append(model.encode(val_user_blocks[curr_layer - 1:], eval_user_features[-1]))	
        eval_news_features.append(model.encode(val_news_blocks[curr_layer - 1:], eval_news_features[-1]))
        
        # message passing the training features for last layer
        final_pos_features = [model.encode(all_pos_blocks[curr_layer - 1:], final_pos_features[-1])]
        final_neg_features = [model.encode(all_neg_blocks[curr_layer - 1:], final_neg_features[-1])]
        
    torch.save(model, '{}/{}_layer{}_seed={}_ckt={}.pth'.format(
            hydra_output_dir, 
            "mind", 
            curr_layer, 
            cfg.seed,
            best_epoch if cfg.best_model else cfg.num_rounds[curr_layer]
        ))
            
    # Last GNN Layer (2)
    curr_layer += 1
    model = setup_model(global_model, curr_layer, cfg.device)
    sync_model(model)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[curr_layer])
    
    best_epoch, best_model = train_gnn_layer(model, dataset, cfg, device, log_dir, curr_layer, opt, final_pos_sample_graph, all_pos_blocks[1:], final_neg_sample_graph, all_neg_blocks[1:], val_user_blocks, val_news_blocks,  eval_user_features[-1],  eval_news_features[-1], final_pos_features[-1], final_neg_features[-1], perf_stores)
    
    for i, perf_store in enumerate(perf_stores):
        print(f"Performance Store {i}:")
        print(f"Grad Reduce Time: {perf_store.get_mean_grad_reduce_times()}, STD: {perf_store.get_std_grad_reduce_times()}")
        print(f"Forward Pass Time: {perf_store.get_mean_forward_pass_time()}, STD: {perf_store.get_std_forward_pass_time()}")
        print(f"Local Grad Encryption: {perf_store.get_mean_grad_encryption()}, STD: {perf_store.get_std_grad_encryption()}")
        print(f"Local Train Time: {perf_store.get_mean_local_train_time()}, STD: {perf_store.get_std_local_train_time()}")
        print(f"Compute Loss Time: {perf_store.get_mean_compute_loss_time()}, STD: {perf_store.get_std_compute_loss_time()}")
        print(f"Backward Pass Time: {perf_store.get_mean_backward_pass_time()}, STD: {perf_store.get_std_backward_pass_time()}")


        
    if cfg.best_model:
        model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        # prepare graph for evaluation
        for ntype in dataset.num_node:
            dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([dataset.num_node[ntype], cfg.hidden_dim * 2]).float()
        for etype in dataset.num_relation:
            dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones([dataset.num_relation[etype]]).float() * 0.5

        dataset.graph.nodes['user'].data['GNN_Emb'][val_user_blocks[-1].dstdata['_ID']['user'].long()] = model.encode(val_user_blocks[curr_layer - 1:], eval_user_features[-1])['user'].cpu()
    
        dataset.graph.nodes['news'].data['GNN_Emb'][val_news_blocks[-1].dstdata['_ID']['news'].long()] = model.encode(val_news_blocks[curr_layer - 1:], eval_news_features[-1])['news'].cpu()
        
        if cfg["quick_eval"]:
            result = quick_rec(model, dataset, cfg.num_rounds[curr_layer], cfg)
        else:
            result = full_rec(model, dataset, cfg.num_rounds[curr_layer], cfg)
        
        wandb.log({
            f"auc": result[0],
            f"mrr": result[1],
            f"ndgc@5": result[2],
            f"ndgc@10": result[3]
            })
        wandb.run.summary["auc"] = result[0]
        wandb.run.summary["mrr"] = result[1]
        wandb.run.summary["ndgc@5"] = result[2]
        wandb.run.summary["ndgc@10"] = result[3]
        
        torch.save(model, '{}/{}_layer{}_seed={}_ckt={}.pth'.format(
            hydra_output_dir, 
            "mind", 
            curr_layer, 
            cfg.seed,
            best_epoch if cfg.best_model else cfg.num_rounds[curr_layer]
        ))
            
    return result, 0, best_epoch

def train_gnn_layer(model, dataset, cfg, device, log_dir, curr_layer, opt, pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks, val_user_blocks, val_news_blocks, val_user_features, val_news_features, pos_features, neg_features, perf_stores):
    best_loss = 1000000
    best_score = 0
    best_auc = 0
    best_model = None
    
    perf_store = PerformanceStore()
    perf_stores.append(perf_store)
    
    with open("public_key.pem", "rb") as public_file:
        public_key = serialization.load_pem_public_key(
            public_file.read(),
            backend=default_backend()
        )
    reducer = MultiThreadReducerCentralized(
        model, cfg.sleep_time, public_key, perf_store, cfg.measure_dv, 
    )
  
    print(f"Training Layer {curr_layer}...")
    for i in range(cfg.num_rounds[curr_layer]):
        model.train()
        
        forward_pass_s = time.time()

        pos_scores, pos_output_features, pos_gnn_kls = model(pos_sample_graph, pos_blocks, ('user', 'pos_train', 'news'), pos_features)
        neg_scores, neg_output_features, neg_gnn_kls = model(neg_sample_graph, neg_blocks, ('user', 'neg_train', 'news'), neg_features)
        
        perf_store.add_forward_pass_time(time.time() - forward_pass_s)
        compute_loss_s = time.time()

        pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.reshape(-1, cfg['gnn_neg_ratio'])], dim=1)
        score_diff = (F.sigmoid(pred)[:, 0] - F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()
        
        if cfg['loss_func'] == 'log_sofmax':
            pred_loss = (-torch.log_softmax(pred, dim=1).select(1, 0)).mean()
        elif cfg['loss_func'] == 'cross_entropy':
            label = torch.cat([torch.ones([pred.shape[0], 1]), torch.zeros([pred.shape[0], cfg['gnn_neg_ratio']])], dim=1).to(device)
            pred_loss = F.binary_cross_entropy(F.sigmoid(pred), label)
        else:
            raise Exception('Unexpected Loss Function')

        loss = pred_loss 
        wandb.log({f"train loss": loss}, step=sum(cfg.num_rounds[:curr_layer]) + (i + 1))
        wandb.log({f"score diff": score_diff}, step=sum(cfg.num_rounds[:curr_layer]) + (i + 1))
        perf_store.add_compute_loss_time(time.time() - compute_loss_s)
        backward_pass_s = time.time()
        
        opt.zero_grad()
        loss.backward()
        
        perf_store.add_backward_pass_time(time.time() - backward_pass_s)
        
        with torch.no_grad():
            aggr_time_s = time.time()
            if cfg.enclave:
                reducer.secure_aggregation(model, ["user, news"], i, perf_store)
            else:
                reducer.aggregate_plaingrads(model, ["user, news"], i, perf_store)
                
        aggr_time = time.time() - aggr_time_s
        perf_store.add_grad_reduce_time(aggr_time)

        opt.step()
        
        perf_store.add_local_train_time(time.time() - forward_pass_s)

        if  (i + 1) % cfg.log_every == 0:
            if cfg.print_all:
                print('\nTrain Result Layer {} @ Iter = {}\n- Training Loss = {}\n- Predict Loss = {}\n- \n- Score Diff = {}\n'.format(
                    curr_layer, i, loss.item(), pred_loss.item(), score_diff.item()
                ))
            with open(log_dir + "/losses_1.txt", "a+") as f:
                f.write(
                    f'{curr_layer}:{loss}:{score_diff}\n'
                )
                
        if  i > cfg.eval_after[curr_layer] and (i + 1) % cfg.eval_every[curr_layer] == 0:
            model.eval()
            print(f"Evaluating epoch {i}...")
            with torch.no_grad():             
                # prepare graph for evaluation
                for ntype in dataset.num_node:
                    dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([dataset.num_node[ntype], cfg.hidden_dim * 2]).float()
                for etype in dataset.num_relation:
                    dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones([dataset.num_relation[etype]]).float() * 0.5

                dataset.graph.nodes['user'].data['GNN_Emb'][val_user_blocks[-1].dstdata['_ID']['user'].long()] = model.encode(val_user_blocks[-1:], val_user_features, for_prediction=True)['user'].cpu()
    
                dataset.graph.nodes['news'].data['GNN_Emb'][val_news_blocks[-1].dstdata['_ID']['news'].long()] = model.encode(val_news_blocks[-1:], val_news_features, for_prediction=True)['news'].cpu()
                        
                result = quick_eval(model, dataset, i, cfg)
            
            wandb.log({
            f"auc": result[0],
            f"mrr": result[1],
            f"ndgc@5": result[2],
            f"ndgc@10": result[3]
            }, step=sum(cfg.num_rounds[:curr_layer]) + i + 1)
            
            if result[0] > best_auc:
                best_auc = result[0]
                best_loss = loss
                best_score = score_diff
                best_epoch = i
                best_model = deepcopy(model.state_dict())     
                
    if best_model is None:
        best_loss = loss
        best_score = score_diff
        best_epoch = i
        best_model = deepcopy(model.state_dict())
    
     
    fstr = f'Ending layer {curr_layer} after {i} rounds with auc {best_auc} loss {best_loss} and score {best_score} (round {best_epoch})\n'
    print(fstr)
    with open(log_dir + "/early_stop.txt", "a+") as f:
        f.write(fstr)
    return best_epoch, best_model

def train_embedding_layer(model, dataset, cfg, device, log_dir, opt, pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks, val_user_blocks, val_news_blocks, perf_stores):
    best_loss = 1000000
    best_score = 0
    best_auc = 0
    curr_layer = 0
    best_model = None
    perf_store = perf_stores[0]
    
    with open("public_key.pem", "rb") as public_file:
        public_key = serialization.load_pem_public_key(
            public_file.read(),
            backend=default_backend()
        )
    reducer = MultiThreadReducerCentralized(
        model, cfg.sleep_time, public_key, comm_volume_perf_store, cfg.measure_dv, 
    )
        
    print("Starting training retexo, Layer 0...")
    for i in range(cfg.num_rounds[0]):
        model.train()
        
        forward_pass_s = time.time()
        
        pos_sample_graph = pos_sample_graph.to(device)
        pos_blocks = [b.to(device) for b in pos_blocks]
        pos_scores, pos_output_features, _ = model(pos_sample_graph, pos_blocks, ('user', 'pos_train', 'news'))
        neg_sample_graph = neg_sample_graph.to(device)
        neg_blocks = [b.to(device) for b in neg_blocks]
        neg_scores, neg_output_features, _ = model(neg_sample_graph, neg_blocks, ('user', 'neg_train', 'news'))
        
        perf_store.add_forward_pass_time(time.time() - forward_pass_s)
        compute_loss_s = time.time()
        
        pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.reshape(-1, cfg['gnn_neg_ratio'])], dim=1)
        score_diff = (F.sigmoid(pred)[:, 0] - F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()
        
        if cfg['loss_func'] == 'log_sofmax':
            pred_loss = (-torch.log_softmax(pred, dim=1).select(1, 0)).mean()
        elif cfg['loss_func'] == 'cross_entropy':
            label = torch.cat([torch.ones([pred.shape[0], 1]), torch.zeros([pred.shape[0], cfg['gnn_neg_ratio']])], dim=1).to(device)
            pred_loss = F.binary_cross_entropy(F.sigmoid(pred), label)
        else:
            raise Exception('Unexpected Loss Function')

        loss = pred_loss 
        wandb.log({f"train loss": loss}, step=(i + 1))
        wandb.log({f"score diff": score_diff}, step=(i + 1))
        perf_store.add_compute_loss_time(time.time() - compute_loss_s)
        backward_pass_s = time.time()
        
        opt.zero_grad()
        loss.backward()
        
        perf_store.add_backward_pass_time(time.time() - backward_pass_s)
        
        num_nodes = {}
        for t in ["user", "news"]:
            # TODO: adapt once partitioned dataset is used
            num_nodes[t] = {
                "local": dataset.num_node[t],
                "total": dataset.num_node[t],
            }
        
        # print("secure aggr")
        with torch.no_grad():
            aggr_time_s = time.time()
            if cfg.enclave:
                reducer.secure_aggregation(model, ["user, news"], i, perf_store)
            else:
                reducer.aggregate_plaingrads(model, ["user, news"], i, perf_store)
                
        aggr_time = time.time() - aggr_time_s
        perf_store.add_grad_reduce_time(aggr_time)
           
        # print("step")
        opt.step()
        
        perf_store.add_local_train_time(time.time() - forward_pass_s)
        
        if  (i + 1) % cfg.log_every == 0:
            if cfg.print_all:
                print('\nTrain Result Layer {} @ Iter = {}\n- Training Loss = {}\n- Predict Loss = {}\n- \n- Score Diff = {}\n'.format(
                    curr_layer, i, loss.item(), pred_loss.item(), score_diff.item()
                ))
            with open(log_dir + "/losses_0.txt", "a+") as f:
                f.write(
                    f'{curr_layer}:{loss}:{score_diff}\n'
                )
                
        if  i > cfg.eval_after[curr_layer] and (i + 1) % cfg.eval_every[curr_layer] == 0:
            model.eval()
            print(f"Evaluating epoch {i}...")
            with torch.no_grad():        
                # prepare graph for evaluation
                for ntype in dataset.num_node:
                    dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([dataset.num_node[ntype], cfg.hidden_dim * 2]).float()
                for etype in dataset.num_relation:
                    dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones([dataset.num_relation[etype]]).float() * 0.5

                dataset.graph.nodes['user'].data['GNN_Emb'][val_user_blocks[-1].dstdata['_ID']['user'].long()] = model.encode([val_user_blocks[-1]], for_prediction=True)['user'].cpu()
                dataset.graph.nodes['news'].data['GNN_Emb'][val_news_blocks[-1].dstdata['_ID']['news'].long()] = model.encode([val_news_blocks[-1]], for_prediction=True)['news'].cpu()
                
                result = quick_eval(model, dataset, i, cfg)
            
            wandb.log({
            f"auc": result[0],
            f"mrr": result[1],
            f"ndgc@5": result[2],
            f"ndgc@10": result[3]
            }, step=(i + 1))
            
            if result[0] > best_auc:
                best_auc = result[0]
                best_loss = loss
                best_score = score_diff
                best_epoch = i
                best_model = deepcopy(model.state_dict())
                
    if best_model is None:
        best_loss = loss
        best_score = score_diff
        best_epoch = i
        best_model = deepcopy(model.state_dict())
        
    fstr = f'Ending layer {curr_layer} after {i} rounds with auc {best_auc} loss {best_loss} and score {best_score} (round {best_epoch})\n'
    print(fstr)
    with open(log_dir + "/early_stop.txt", "a+") as f:
        f.write(fstr)
    return best_epoch, best_model
        
def train_end2end(model, dataset, cfg, device, log_dir, hydra_output_dir, base_etypes):
    best_acc = 0.0
    best_metrics = []
    best_epoch = 0
    best_model = None
    not_improved_count = 0
    acc_training_time = 0
    global_ct = []
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    
    pos_dataloader, neg_dataloader = dataset.get_gnn_train_loader(base_etypes, cfg["num_layers"])
    assert(len(pos_dataloader) == len(neg_dataloader) == 1)
    pos_input_nodes, pos_sample_graph, pos_blocks = next(iter(pos_dataloader))
    neg_input_nodes, neg_sample_graph, neg_blocks = next(iter(neg_dataloader))
    
    print("Starting training End-2-End...")
    for i in range(cfg.num_rounds[0]):
        model.train()
        forward_pass_s = time.time()

        iter_start_time = time.time()
        pos_sample_graph = pos_sample_graph.to(device)
        pos_blocks = [b.to(device) for b in pos_blocks]
        pos_scores, pos_output_features, pos_gnn_kls = model(pos_sample_graph, pos_blocks, ('user', 'pos_train', 'news'))
        neg_sample_graph = neg_sample_graph.to(device)
        neg_blocks = [b.to(device) for b in neg_blocks]
        neg_scores, neg_output_features, neg_gnn_kls = model(neg_sample_graph, neg_blocks, ('user', 'neg_train', 'news'))

        pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.reshape(-1, cfg['gnn_neg_ratio'])], dim=1)
        score_diff = (F.sigmoid(pred)[:, 0] - F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()
        
        if cfg['loss_func'] == 'log_sofmax':
            pred_loss = (-torch.log_softmax(pred, dim=1).select(1, 0)).mean()
        elif cfg['loss_func'] == 'cross_entropy':
            label = torch.cat([torch.ones([pred.shape[0], 1]), torch.zeros([pred.shape[0], cfg['gnn_neg_ratio']])], dim=1).to(device)
            pred_loss = F.binary_cross_entropy(F.sigmoid(pred), label)
        else:
            raise Exception('Unexpected Loss Function')
        
        # gnn_kl = (sum(pos_gnn_kls) / len(pos_gnn_kls) + sum(neg_gnn_kls) / len(neg_gnn_kls)).mean()
        
        loss = pred_loss #+ cfg['gnn_kl_weight'] * gnn_kl

        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # try other values?
        # if cfg['pruning']:
        #     model.collecting_metapath_utility()
        opt.step()

        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time - iter_start_time
        epoch_time = iter_elapsed_time * len(pos_dataloader)
        acc_training_time += iter_elapsed_time


        if i >= cfg.eval_after[0] and (i + 1) % cfg.log_every[0] == 0:
            print('\nTrain Result @ Iter = {}\n- Training Loss = {}\n- Predict Loss = {}\n- Score Diff = {}\n'.format(
                i, loss.item(), pred_loss.item(), score_diff.item()
            ))
            result = eval(base_etypes, dataset,  cfg.hidden_dim, device, model, i, cfg)
            this_acc = result[0]
            with open(log_dir + "/accuracy.txt", "a+") as f:
                f.write(
                    f'Epoch {i}, AUC: {this_acc} - MRR = {result[1]} - nDCG@5 = {result[2]} - nDCG@10 = {result[3]} - ILAD@5 = {result[4]} - ILAD@10 = {result[5]}\n'
                )
            if this_acc > best_acc:
                best_acc = this_acc
                best_metrics = result
                best_epoch = i
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, '{}/{}_{}_seed={}.pth'.format(
                    hydra_output_dir, 
                    "mind", 
                    "0", 
                    cfg.seed
                ))
                not_improved_count = 0
            else:
                not_improved_count += 1
                if not_improved_count >= cfg.early_stop:
                    break

    return best_metrics, i, best_epoch
        

def eval(etypes, mind_dgl, out_dim, device, model, epoch, cfg):
    for ntype in mind_dgl.num_node:
        mind_dgl.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([mind_dgl.num_node[ntype], out_dim * 2]).float()
    for etype in mind_dgl.num_relation:
        mind_dgl.graph.edges[etype].data['Sampling_Weight'] = torch.ones([mind_dgl.num_relation[etype]]).float() * 0.5

    encode_all_graph(model, mind_dgl, device, etypes)
    
    if cfg["quick_eval"]:
        result = quick_rec(model, mind_dgl, epoch, cfg)
    else:
        result = full_rec(model, mind_dgl, epoch, cfg)
    return result

def init_master(cfg, hydra_output_dir):
    """"Initialize the master process"""
    
    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = cfg.master_port
    dist.init_process_group(backend='gloo', rank=0, world_size=cfg.num_partitions+1)
    
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
    
    # oversee gradient aggregation
    for layer in range(len(cfg.num_rounds)):
        model = setup_model(global_model, layer, "cpu")
        
        sync_model(model)
        reducer = MultiThreadReducerCentralized(model=model) # TODO: add parameters and model
        
        print("Starting layer {}...".format(layer))
        reducer.master_aggregate_gradients(cfg, layer)
        print("Layer {} finished.".format(layer))
            
def init_process(rank, world_size, cfg, hydra_output_dir):
    """Initialize the distributed environment"""
    
    addr = cfg.master_addr
    port = cfg.master_port

    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['LOGLEVEL'] = 'INFO'
    
    dist.init_process_group(
        cfg.distributed.backend, 
        init_method=f'tcp://{addr}:{port}',
        rank=rank, 
        world_size=world_size
    )

    print("Connection established with master node.")
    seed_everything(cfg.seed)
    
    # central version for now
    graph, dataset = load_data(**cfg.dataset.download, cfg=cfg)


    os.makedirs("results/", exist_ok=True)
    os.makedirs(
        f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/",
        exist_ok=True,
    )
    results_dir = f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/rank_{rank}/"
    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    
    
    start_time = time.time()
    train(
        graph,
        dataset,
        cfg,
        hydra_output_dir,
        results_dir,
    )
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")

    dist.destroy_process_group()
    
def encode_all_graph(model, mind_dgl, device, etypes, attention_head=4):
    print('Generating GNN Representation')
    model.eval()
    with torch.no_grad():
        user_dataloader, news_dataloader = mind_dgl.get_gnn_dev_node_loader(etypes, model.n_layers)

        user_dataloader = enumerate(user_dataloader)
        news_dataloader = enumerate(news_dataloader)

        for i, (user_input_nodes, user_sample_graph, user_blocks) in user_dataloader:
            user_blocks = [b.to(device) for b in user_blocks]
            user_output_features = model.encode(user_blocks)

            mind_dgl.graph.nodes['user'].data['GNN_Emb'][user_blocks[-1].dstdata['_ID']['user'].long()] = user_output_features['user'].cpu()
        for i, (news_input_nodes, news_sample_graph, news_blocks) in news_dataloader:
            news_blocks = [b.to(device) for b in news_blocks]
            news_output_features = model.encode(news_blocks)

            mind_dgl.graph.nodes['news'].data['GNN_Emb'][news_blocks[-1].dstdata['_ID']['news'].long()] = news_output_features['news'].cpu()
    print('Generating GNN Representation Finished')
    
    
def full_rec(model, mind_dgl, epoch, cfg):    
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=False)
    cache_size = cfg.cache_size 
    
    
    mind_dgl.graph.nodes['user'].data['News_Pref'] = mind_dgl.graph.nodes['user'].data['GNN_Emb'].unsqueeze(1).unsqueeze(1).repeat(1, mind_dgl.graph.nodes['news'].data['CateID'].max()+1, cache_size, 1)
    mind_dgl.graph.nodes['user'].data['Last_Update_Time'] = torch.zeros([mind_dgl.num_node['user'], mind_dgl.graph.nodes['news'].data['CateID'].max()+1, cache_size])
   
    
    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0


    devloader = tqdm(enumerate(dev_session_loader))

    print("Evaluating...\n")
    for i, (pos_links, neg_links) in devloader:
        if cfg['gnn_quick_dev_reco'] and i >= cfg['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        # sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg_edc, model.scorer.reduce_score_neg_edc, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos_edc, model.scorer.reduce_score_pos_edc, etype=('news', 'pos_dev_r', 'user'))
        mind_dgl.graph.nodes['user'].data['News_Pref'][sub_g.dstdata['_ID']['user'].long()] = sub_g.dstdata['pref']['user']  # write back to mind.graph
        mind_dgl.graph.nodes['user'].data['Last_Update_Time'][sub_g.dstdata['_ID']['user'].long()] = sub_g.dstdata['lut']['user']

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)
        
        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

    if cfg['gnn_quick_dev_reco']:
        epoch_auc_score /= cfg['gnn_quick_dev_reco_size']
        epoch_mrr /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= cfg['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(dev_session_loader)
        epoch_mrr /= len(dev_session_loader)
        epoch_ndcg_5 /= len(dev_session_loader)
        epoch_ndcg_10 /= len(dev_session_loader)
        epoch_ilad_5 /= len(dev_session_loader)
        epoch_ilad_10 /= len(dev_session_loader)

    print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]



def quick_eval(model, mind_dgl, epoch, cfg):
    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0
    
    # create validation samples
    val_session_loader = mind_dgl.get_val_session_loader(cfg.val_size)
    loader = tqdm(enumerate(val_session_loader))

    for i, (pos_links, neg_links) in loader:
        if cfg['gnn_quick_dev_reco'] and i >= cfg['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg, model.scorer.reduce_score_neg, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos, model.scorer.reduce_score_pos, etype=('news', 'pos_dev_r', 'user'))

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)
        
        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

    if cfg['gnn_quick_dev_reco']:
        epoch_auc_score /= cfg['gnn_quick_dev_reco_size']
        epoch_mrr /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= cfg['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(val_session_loader)
        epoch_mrr /= len(val_session_loader)
        epoch_ndcg_5 /= len(val_session_loader)
        epoch_ndcg_10 /= len(val_session_loader)
        epoch_ilad_5 /= len(val_session_loader)
        epoch_ilad_10 /= len(val_session_loader)
            

    # print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]

def quick_rec(model, mind_dgl, epoch, cfg):
    # testing performance w/o EDC using randomly sampled users
    
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=True)
    cache_size = cfg.cache_size 

    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0


    devloader = enumerate(dev_session_loader)
    # pos_links = mind_dgl._dev_session_positive
    # neg_links = mind_dgl._dev_session_negative

    print("Evaluating...\n")
    for i, (pos_links, neg_links) in devloader:
        if cfg['gnn_quick_dev_reco'] and i >= cfg['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg, model.scorer.reduce_score_neg, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos, model.scorer.reduce_score_pos, etype=('news', 'pos_dev_r', 'user'))

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)
        
        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

    if cfg['gnn_quick_dev_reco']:
        epoch_auc_score /= cfg['gnn_quick_dev_reco_size']
        epoch_mrr /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= cfg['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(dev_session_loader)
        epoch_mrr /= len(dev_session_loader)
        epoch_ndcg_5 /= len(dev_session_loader)
        epoch_ndcg_10 /= len(dev_session_loader)
        epoch_ilad_5 /= len(dev_session_loader)
        epoch_ilad_10 /= len(dev_session_loader)
            

    print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]


