"""Implement trainer that oversees end-to-end training process"""

import pickle
import wandb
import os
import logging
from copy import deepcopy
import time
import dgl  # type: ignore
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler, Adam
import torch.distributed as dist

from omegaconf import DictConfig
from data.dataset import load_data

from models.custom_divhgnn import CustomDivHGNN
from models.utils.layers import seed_everything
from trainers.metrics import auc, mrr, nDCG, compute_loss

import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


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
) -> None:
    """Implement end-to-end training process"""
    # set the seed
    set_torch_seed(cfg.seed)
    base_etypes = ['history', 'history_r', 'ne_link',
                   'ne_link_r', 'ue_link', 'ue_link_r']
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

    wandb.init(project=f"{cfg.app}-{cfg.dataset_name}",
               config=dict(cfg), mode=cfg.wandb_mode)

    for ntype in dataset.num_node:
        dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros(
            [dataset.num_node[ntype], cfg['hidden_dim'] * 2]).float()
    for etype in dataset.num_relation:
        dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones(
            [dataset.num_relation[etype]]).float() * 0.5

    base_canonical_etypes = sorted(
        [canonical_etype for canonical_etype in dataset.graph.canonical_etypes if canonical_etype[1] in base_etypes])

    # set up the model
    device = cfg.device
    # rank, _ = dist.get_rank(), dist.get_world_size()

    log_dir = os.path.join(hydra_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Model
    num_layers = cfg.num_layers
    model = CustomDivHGNN(
        cfg.adaptor_hidden, cfg.gnn_input_dim, cfg.hidden_dim, cfg.hidden_dim, node_emb_meta, device, num_layers, cfg.cross_score, cfg.dropout
    ).to(device)

    perf_metrics, epoch, best_epoch = train_retexo(
        model, dataset, cfg, device, log_dir, hydra_output_dir, base_etypes)
    fstr = '\nDONE after {} iterations\nBest AUC: {} at epoch {}. All metrics: {}'.format(
        epoch, perf_metrics[0], best_epoch, perf_metrics)
    print(fstr)
    with open(log_dir + "/accuracy.txt", "a+") as f:
        f.write(fstr)


def train_retexo(model, dataset, cfg, device, log_dir, hydra_output_dir, base_etypes):
    curr_layer = 0
    acc_training_time = 0
    not_improved = 0
    global_model = model

    model = setup_model(global_model, curr_layer, cfg.device)
    opt = Adam(model.parameters(), lr=cfg.learning_rate[0])

    # all train blocks (for layer-wise forward passes)
    full_pos_dataloader, full_neg_dataloader = dataset.get_gnn_train_loader(
        base_etypes, 2, batching=cfg.batching)

    # create the full batch of training edges
    pos_dataloader, neg_dataloader = dataset.get_gnn_train_loader(
        base_etypes, 1, batching=cfg.batching)

    if not cfg.batching:
        assert (len(full_pos_dataloader) == len(full_neg_dataloader) == 1)
        _, final_pos_sample_graph, all_pos_blocks = next(
            iter(full_pos_dataloader))
        _, final_neg_sample_graph, all_neg_blocks = next(
            iter(full_neg_dataloader))
        assert (len(pos_dataloader) == len(neg_dataloader) == 1)
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
    user_dataloader, news_dataloader = dataset.get_gnn_dev_node_loader(
        base_etypes, cfg.num_layers)
    (__annotations__, _, user_blocks) = next(iter(user_dataloader))
    val_user_blocks = [b.to(device) for b in user_blocks]
    (_, _, news_blocks) = next(iter(news_dataloader))
    val_news_blocks = [b.to(device) for b in news_blocks]

    del user_dataloader
    del news_dataloader

    if cfg.batching:
        best_epoch, best_model = train_batched_layer(
            model, dataset, cfg, device, opt,
            pos_dataloader, neg_dataloader, val_user_blocks, val_news_blocks,
            None, None, 0, None, None)
    else:
        best_epoch, best_model = train_layer(
            model, dataset, cfg, device, log_dir, curr_layer, opt,
            pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks,
            val_user_blocks, val_news_blocks, None, None, None, None)

    print("Layer 0 training finished, encoding features for next layer...")
    if cfg.best_model:
        model.load_state_dict(best_model)

    # Layer is frozen. Encode features for next layer
    model.eval()
    with torch.no_grad():
        # encoding the eval features
        eval_user_features = [model.encode(
            val_user_blocks, encode_source=True)]
        eval_news_features = [model.encode(
            val_news_blocks, encode_source=True)]

        middle_user_features = model.encode(
            val_user_blocks[1:], encode_source=True)
        middle_news_features = model.encode(
            val_news_blocks[1:], encode_source=True)

        if not cfg.batching:
            # encoding the training features for last layer
            final_pos_sample_graph = final_pos_sample_graph.to(device)
            all_pos_blocks = [b.to(device) for b in all_pos_blocks]
            final_pos_features = [model.encode(
                all_pos_blocks, encode_source=True)]
            final_neg_sample_graph = final_neg_sample_graph.to(device)
            all_neg_blocks = [b.to(device) for b in all_neg_blocks]
            final_neg_features = [model.encode(
                all_neg_blocks, encode_source=True)]

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
    embedding_model = model
    model = setup_model(global_model, curr_layer, cfg.device)
    opt = Adam(model.parameters(), lr=cfg.learning_rate[curr_layer])

    if cfg.batching:
        best_epoch, best_model = train_batched_layer(
            model, dataset, cfg, device, opt,
            pos_dataloader, neg_dataloader, val_user_blocks, val_news_blocks,
            middle_user_features, middle_news_features,
            curr_layer, embedding_model, None)
    else:
        best_epoch, best_model = train_layer(
            model, dataset, cfg, device, log_dir, curr_layer, opt,
            pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks,
            val_user_blocks, val_news_blocks, middle_user_features, middle_news_features,
            pos_features, neg_features)

    if cfg.best_model:
        model.load_state_dict(best_model)

    # Layer is frozen, encoding features for last layer
    model.eval()
    with torch.no_grad():
        # message passing eval features for last layer
        eval_user_features.append(model.encode(
            val_user_blocks[curr_layer - 1:], eval_user_features[-1]))
        eval_news_features.append(model.encode(
            val_news_blocks[curr_layer - 1:], eval_news_features[-1]))

        if not cfg.batching:
            # message passing the training features for last layer
            final_pos_features = [model.encode(
                all_pos_blocks[curr_layer - 1:], final_pos_features[-1])]
            final_neg_features = [model.encode(
                all_neg_blocks[curr_layer - 1:], final_neg_features[-1])]

    torch.save(model, '{}/{}_layer{}_seed={}_ckt={}.pth'.format(
        hydra_output_dir,
        "mind",
        curr_layer,
        cfg.seed,
        best_epoch if cfg.best_model else cfg.num_rounds[curr_layer]
    ))

    # Last GNN Layer (2)
    curr_layer += 1
    middle_model = model
    model = setup_model(global_model, curr_layer, cfg.device)
    opt = Adam(model.parameters(), lr=cfg.learning_rate[curr_layer])

    if cfg.batching:
        best_epoch, best_model = train_batched_layer(
            model, dataset, cfg, device, opt, full_pos_dataloader, full_neg_dataloader, val_user_blocks, val_news_blocks, eval_user_features[-1], eval_news_features[-1], curr_layer, embedding_model, middle_model)
    else:
        best_epoch, best_model = train_layer(
            model, dataset, cfg, device, log_dir, curr_layer, opt,
            final_pos_sample_graph, all_pos_blocks[1:],
            final_neg_sample_graph, all_neg_blocks[1:],
            val_user_blocks, val_news_blocks,  eval_user_features[-1],  eval_news_features[-1],
            final_pos_features[-1], final_neg_features[-1])

    if cfg.best_model:
        model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        # prepare graph for evaluation
        for ntype in dataset.num_node:
            dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros(
                [dataset.num_node[ntype], cfg.hidden_dim * 2]).float()
        for etype in dataset.num_relation:
            dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones(
                [dataset.num_relation[etype]]).float() * 0.5

        dataset.graph.nodes['user'].data['GNN_Emb'][val_user_blocks[-1].dstdata['_ID']['user'].long(
        )] = model.encode(val_user_blocks[curr_layer - 1:], eval_user_features[-1])['user'].cpu()

        dataset.graph.nodes['news'].data['GNN_Emb'][news_blocks[-1].dstdata['_ID']['news'].long(
        )] = model.encode(val_news_blocks[curr_layer - 1:], eval_news_features[-1])['news'].cpu()

        if cfg["quick_eval"]:
            result = quick_eval(
                model, dataset, cfg.num_rounds[curr_layer], cfg)
        else:
            result = full_rec(model, dataset, cfg.num_rounds[curr_layer], cfg)

        wandb.log({
            f"auc": result[0],
            f"mrr": result[1],
            f"ndgc@5": result[2],
            f"ndgc@10": result[3]
            # f"val_loss": result[6]
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

# def train_gnn_layer(model, dataset, cfg, device, log_dir, curr_layer, opt, pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks, val_user_blocks, val_news_blocks, val_user_features, val_news_features, pos_features, neg_features):


def train_layer(model, dataset, cfg, device, log_dir, curr_layer, opt, pos_sample_graph, pos_blocks, neg_sample_graph, neg_blocks, val_user_blocks, val_news_blocks, val_user_features, val_news_features, pos_features, neg_features):
    best_loss = 1000000
    best_score = 0
    best_auc = 0
    best_ndgc10 = 0
    training_times = []

    if cfg.exp_lr:
        scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9997712)

    for i in range(cfg.num_rounds[curr_layer]):
        model.train()

        pos_sample_graph = pos_sample_graph.to(device)
        pos_blocks = [b.to(device) for b in pos_blocks]

        neg_sample_graph = neg_sample_graph.to(device)
        neg_blocks = [b.to(device) for b in neg_blocks]

        pos_scores, pos_output_features, pos_gnn_kls = model(
            pos_sample_graph, pos_blocks, ('user', 'pos_train', 'news'), pos_features)
        neg_scores, neg_output_features, neg_gnn_kls = model(
            neg_sample_graph, neg_blocks, ('user', 'neg_train', 'news'), neg_features)

        pred = torch.cat([pos_scores.unsqueeze(
            1), neg_scores.reshape(-1, cfg['gnn_neg_ratio'])], dim=1)
        score_diff = (F.sigmoid(pred)[:, 0] -
                      F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()

        loss = compute_loss(cfg, device, pred)
        wandb.log({f"train loss": loss}, step=sum(
            cfg.num_rounds[:curr_layer]) + (i))
        wandb.log({f"score diff": score_diff}, step=sum(
            cfg.num_rounds[:curr_layer]) + (i))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if cfg.exp_lr:
            scheduler.step()

        if i > cfg.eval_after[curr_layer] and (i + 1) % cfg.eval_every[curr_layer] == 0:
            model.eval()
            with torch.no_grad():
                # prepare graph for evaluation
                for ntype in dataset.num_node:
                    dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros(
                        [dataset.num_node[ntype], cfg.hidden_dim * 2]).float()
                for etype in dataset.num_relation:
                    dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones(
                        [dataset.num_relation[etype]]).float() * 0.5

                user_ids = val_user_blocks[-1].dstdata['_ID']['user'].long()
                user_features = None if curr_layer == 0 else val_user_features
                user_data = model.encode(
                    val_user_blocks[-1:], user_features, for_prediction=True)['user'].cpu()

                news_ids = val_news_blocks[-1].dstdata['_ID']['news'].long()
                news_features = None if curr_layer == 0 else val_news_features
                news_data = model.encode(
                    val_news_blocks[-1:], news_features, for_prediction=True)['news'].cpu()

                dataset.graph.nodes['user'].data['GNN_Emb'][user_ids] = user_data
                dataset.graph.nodes['news'].data['GNN_Emb'][news_ids] = news_data

                result = quick_eval(model, dataset, i, cfg)

            wandb.log({
                f"auc": result[0],
                f"mrr": result[1],
                f"ndgc@5": result[2],
                f"ndgc@10": result[3],
                f"val_loss": result[6]
            }, step=sum(cfg.num_rounds[:curr_layer]) + i)

            if result[0] > best_auc:
                best_auc = result[0]
                best_ndgc10 = result[3]
                best_loss = loss
                best_score = score_diff
                best_epoch = i
                best_model = deepcopy(model.state_dict())

    fstr = f'Ending layer {curr_layer} after {i} rounds with auc {best_auc} loss {best_loss} and score {best_score} (round {best_epoch})\n'
    print(fstr)
    return best_epoch, best_model



def train_batched_layer(model, dataset, cfg, device, opt, pos_dataloader, neg_dataloader, val_user_blocks, val_news_blocks, val_user_features, val_news_features, layer, embedding_model, middle_model):
    pos_dataloader = iter(pos_dataloader)
    neg_dataloader = iter(neg_dataloader)
    best_loss = 1000000
    best_score = 0
    best_auc = 0
    best_ndgc10 = 0
    training_times = []

    if cfg.exp_lr:
        scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9997712)

    for i in range(cfg.num_rounds[layer]):
        model.train()

        # Random sampling
        _, pos_sample_graph, pos_blocks = next(pos_dataloader)
        _, neg_sample_graph, neg_blocks = next(neg_dataloader)
        pos_sample_graph = pos_sample_graph.to(device)
        pos_blocks = [b.to(device) for b in pos_blocks]
        neg_sample_graph = neg_sample_graph.to(device)
        neg_blocks = [b.to(device) for b in neg_blocks]

        pos_features, neg_features = None, None
        with torch.no_grad():
            if embedding_model is not None:
                pos_features = embedding_model.encode(
                    pos_blocks, encode_source=True)
                neg_features = embedding_model.encode(
                    neg_blocks, encode_source=True)

            if middle_model is not None:
                pos_features = middle_model.encode(pos_blocks, pos_features)
                neg_features = middle_model.encode(neg_blocks, neg_features)

        # Forward pass on positive and negative samples
        pos_scores, _, _ = model(
            pos_sample_graph, pos_blocks if middle_model is None else pos_blocks[1:], ('user', 'pos_train', 'news'), pos_features)

        neg_scores, _, _ = model(
            neg_sample_graph, neg_blocks if middle_model is None else neg_blocks[1:], ('user', 'neg_train', 'news'), neg_features)

        # Score and loss computation
        pred = torch.cat([pos_scores.unsqueeze(
            1), neg_scores.reshape(-1, cfg['gnn_neg_ratio'])], dim=1)
        score_diff = (F.sigmoid(pred)[:, 0] -
                      F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()

        loss = compute_loss(cfg, device, pred)
        wandb.log({f"train loss": loss},
                  step=sum(cfg.num_rounds[:layer]) + (i))
        wandb.log({f"score diff": score_diff},
                  step=sum(cfg.num_rounds[:layer]) + (i))
        if cfg.exp_lr:
            wandb.log({f"LearR": scheduler.get_last_lr()[0]},
                      step=sum(cfg.num_rounds[:layer]) + (i))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if cfg.exp_lr:
            scheduler.step()

        # Periodic evaluation
        if (i + 1) > cfg.eval_after[layer] and (i + 1) % cfg.eval_every[layer] == 0:
            model.eval()
            with torch.no_grad():
                # prepare graph for evaluation
                for ntype in dataset.num_node:
                    dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros(
                        [dataset.num_node[ntype], cfg.hidden_dim * 2]).float()
                for etype in dataset.num_relation:
                    dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones(
                        [dataset.num_relation[etype]]).float() * 0.5

                val_user_blocks = [val_user_blocks[-1]] if layer == 0 else val_user_blocks[-1:]
                val_news_blocks =  [val_news_blocks[-1]] if layer == 0 else val_news_blocks[-1:]

                dataset.graph.nodes['user'].data['GNN_Emb'][val_user_blocks[-1].dstdata['_ID']['user'].long(
                )] = model.encode(val_user_blocks, val_user_features,  for_prediction=True)['user'].cpu()
                dataset.graph.nodes['news'].data['GNN_Emb'][val_news_blocks[-1].dstdata['_ID']['news'].long(
                )] = model.encode(val_news_blocks, val_news_features, for_prediction=True)['news'].cpu()

                result = quick_eval(model, dataset, i, cfg)

            wandb.log({
                f"auc": result[0],
                f"mrr": result[1],
                f"ndgc@5": result[2],
                f"ndgc@10": result[3],
                f"val_loss": result[6]
            }, step=sum(cfg.num_rounds[:layer]) + (i))

            # Remember best model
            if result[0] > best_auc:
                best_auc = result[0]
                best_ndgc10 = result[3]
                best_loss = loss
                best_score = score_diff
                best_epoch = i
                best_model = deepcopy(model.state_dict())

    fstr = f'Ending layer {layer} after {i} rounds with auc {best_auc} loss {best_loss} and score {best_score} (round {best_epoch})\n'
    print(fstr)
    return best_epoch, best_model



def init_process(rank, cfg, hydra_output_dir):
    """Initialize the distributed environment"""

    os.environ["MASTER_ADDR"] = cfg.controller_addr
    os.environ["MASTER_PORT"] = cfg.controller_port
    dist.init_process_group(
        cfg.backend, rank=rank, world_size=cfg.num_partitions
    )

    seed_everything(cfg.seed)

    # central version for now
    graph, dataset = load_data(cfg=cfg)

    train(
        graph,
        dataset,
        cfg,
        hydra_output_dir,
    )

    dist.destroy_process_group()

# ====== Evaluation functions for our graph model ======


def encode_all_graph(model, mind_dgl, device, etypes, attention_head=4):
    print('Generating GNN Representation')
    model.eval()
    with torch.no_grad():
        user_dataloader, news_dataloader = mind_dgl.get_gnn_dev_node_loader(
            etypes, model.n_layers)

        user_dataloader = enumerate(user_dataloader)
        news_dataloader = enumerate(news_dataloader)

        for i, (user_input_nodes, user_sample_graph, user_blocks) in user_dataloader:
            user_blocks = [b.to(device) for b in user_blocks]
            user_output_features = model.encode(user_blocks)

            mind_dgl.graph.nodes['user'].data['GNN_Emb'][user_blocks[-1].dstdata['_ID']
                                                         ['user'].long()] = user_output_features['user'].cpu()
        for i, (news_input_nodes, news_sample_graph, news_blocks) in news_dataloader:
            news_blocks = [b.to(device) for b in news_blocks]
            news_output_features = model.encode(news_blocks)

            mind_dgl.graph.nodes['news'].data['GNN_Emb'][news_blocks[-1].dstdata['_ID']
                                                         ['news'].long()] = news_output_features['news'].cpu()
    print('Generating GNN Representation Finished')


def full_rec(model, mind_dgl, epoch, cfg):
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=False)
    cache_size = cfg.cache_size

    mind_dgl.graph.nodes['user'].data['News_Pref'] = mind_dgl.graph.nodes['user'].data['GNN_Emb'].unsqueeze(
        1).unsqueeze(1).repeat(1, mind_dgl.graph.nodes['news'].data['CateID'].max()+1, cache_size, 1)
    mind_dgl.graph.nodes['user'].data['Last_Update_Time'] = torch.zeros(
        [mind_dgl.num_node['user'], mind_dgl.graph.nodes['news'].data['CateID'].max()+1, cache_size])

    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0
    epoch_tp = 0
    epoch_tn = 0
    epoch_fp = 0
    epoch_fn = 0

    devloader = tqdm(enumerate(dev_session_loader))

    print("Evaluating...\n")
    for i, (pos_links, neg_links) in devloader:
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {(
            'news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        # sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg_edc,
                         model.scorer.reduce_score_neg_edc, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos_edc,
                         model.scorer.reduce_score_pos_edc, etype=('news', 'pos_dev_r', 'user'))
        mind_dgl.graph.nodes['user'].data['News_Pref'][sub_g.dstdata['_ID']['user'].long(
        )] = sub_g.dstdata['pref']['user']  # write back to mind.graph
        mind_dgl.graph.nodes['user'].data['Last_Update_Time'][sub_g.dstdata['_ID']['user'].long(
        )] = sub_g.dstdata['lut']['user']

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(
            sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'],
                           sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'],
                                        sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)

        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(),
                      scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(),
                       scores.unsqueeze(0).numpy(), k=10)

        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        predictions = (scores >= 0.5).int()
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        tn = ((predictions == 0) & (labels == 0)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()

        total_positives = tp + fn
        total_negatives = tn + fp

        epoch_tp += tp / total_positives if total_positives > 0 else 0
        epoch_tn += tn / total_negatives if total_negatives > 0 else 0
        epoch_fp += fp / total_negatives if total_negatives > 0 else 0
        epoch_fn += fn / total_positives if total_positives > 0 else 0

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(
                scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (
            top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T

        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(
                scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (
            top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T


    num_batches = len(dev_session_loader)

    epoch_auc_score /= num_batches
    epoch_mrr /= num_batches
    epoch_ndcg_5 /= num_batches
    epoch_ndcg_10 /= num_batches
    epoch_ilad_5 /= num_batches
    epoch_ilad_10 /= num_batches

    avg_tp_ratio = epoch_tp / num_batches
    avg_fp_ratio = epoch_fp / num_batches
    avg_tn_ratio = epoch_tn / num_batches
    avg_fn_ratio = epoch_fn / num_batches

    print('Testing Result @ Epoch = {}\n'
          '- AUC = {}\n'
          '- MRR = {}\n'
          '- nDCG@5 = {}\n'
          '- nDCG@10 = {}\n'
          '- ILAD@5 = {}\n'
          '- ILAD@10 = {}\n'
          '- TP Ratio = {:.4f}\n'
          '- FP Ratio = {:.4f}\n'
          '- TN Ratio = {:.4f}\n'
          '- FN Ratio = {:.4f}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10, avg_tp_ratio, avg_fp_ratio, avg_tn_ratio, avg_fn_ratio,))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]


def cache_edge_results(mind_dgl, val_session_loader, cfg, cache_file):
    cached_results = []
    loader = enumerate(val_session_loader)

    for i, (pos_links, neg_links) in loader:
        pos_links = torch.tensor(pos_links, dtype=torch.int32)
        neg_links = torch.tensor(neg_links, dtype=torch.int32)
        pos_src, pos_dst = mind_dgl.graph.find_edges(
            pos_links, etype=('news', 'pos_dev_r', 'user'))
        neg_src, neg_dst = mind_dgl.graph.find_edges(
            neg_links, etype=('news', 'neg_dev_r', 'user'))

        cached_results.append((pos_src, pos_dst, neg_src, neg_dst))

    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
    print("Cached validation edges saved.")
    return cached_results


def load_cached_results(size, cache_file):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def quick_eval(model, mind_dgl, epoch, cfg):
    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0
    epoch_val_loss = 0

    cache_file = f'{cfg.dataset_dir}/cached_val_edges_{cfg.eval_size}.pkl'
    if os.path.exists(cache_file):
        cached_results = load_cached_results(cfg.eval_size, cache_file)
    else:
        print("Computing and caching validation edges...")
        val_session_loader = mind_dgl.get_val_session_loader(cfg.eval_size)
        cached_results = cache_edge_results(mind_dgl, val_session_loader, cfg, cache_file)

    out_features = model.scorer.out_features
    val_edges = enumerate(cached_results)
    for i, (pos_src, pos_dst, neg_src, neg_dst) in val_edges:
        pos_user_repr = mind_dgl.graph.nodes["user"].data['GNN_Emb'][pos_dst, :out_features]
        neg_user_repr = mind_dgl.graph.nodes["user"].data['GNN_Emb'][neg_dst, :out_features]
        pos_news_repr = mind_dgl.graph.nodes["news"].data['GNN_Emb'][pos_src, :out_features]
        neg_news_repr = mind_dgl.graph.nodes["news"].data['GNN_Emb'][neg_src, :out_features]

        pos_score = (pos_user_repr * pos_news_repr).sum(dim=1)
        neg_score = (neg_user_repr * neg_news_repr).sum(dim=1)

        labels = torch.cat([torch.ones(pos_score.shape), torch.zeros(
            neg_score.shape)], dim=0).type(torch.int32).squeeze(0)
        scores = torch.cat([pos_score, neg_score]).squeeze(0)

        val_loss = F.binary_cross_entropy(
            F.sigmoid(scores), labels.type(torch.float32))
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(),
                      scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(),
                       scores.unsqueeze(0).numpy(), k=10)

        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10
        epoch_val_loss += val_loss

    epoch_auc_score /= len(cached_results)
    epoch_mrr /= len(cached_results)
    epoch_ndcg_5 /= len(cached_results)
    epoch_ndcg_10 /= len(cached_results)
    epoch_ilad_5 /= len(cached_results)
    epoch_ilad_10 /= len(cached_results)
    epoch_val_loss /= len(cached_results)

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10, epoch_val_loss]
