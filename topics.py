import logging
from typing import Dict, List, Optional, Tuple
import torch
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os
from data.dataset import load_data

from data.mind_dataset import MIND_DGL
import dgl # type: ignore
from trainers.metrics import *


def get_top_5_news_per_user(graph):
    # Get the user-news subgraph
    user_news_subgraph = graph.edge_type_subgraph(['history'])
    
    # Get the edge IDs
    user_ids, news_ids = user_news_subgraph.edges(etype='history')
    
    # Get news categories
    news_categories = graph.ndata['SubCategory']['news']
    summed_categories = torch.sum(news_categories, dim=1)
    
    # Create a tensor of (user_id, category)
    user_category_pairs = torch.stack([user_ids, summed_categories[news_ids]], dim=1)
    
    # Count occurrences of each unique (user_id, category) pair
    unique_pairs, counts = torch.unique(user_category_pairs, dim=0, return_counts=True)
    
    user_category_counts = defaultdict(dict)
    for (user_id, category), count in zip(unique_pairs, counts):
        user_id = user_id.item()
        category = category.item()
        user_category_counts[user_id][category] = count.item()
    
    # # Count categories for each user
    # user_category_counts = defaultdict(lambda: defaultdict(int))
    # for user_id, news_id in tqdm(zip(user_ids, news_ids), total=len(user_ids), desc="Processing user-news interactions"):
    #     user_id = user_id.item()
    #     category = sum(news_categories[news_id]).item()
    #     user_category_counts[user_id][category] += 1
    
    # Get top 5 categories for each user
    top_5_per_user = {}
    for user_id in tqdm(user_category_counts.keys(), desc="Computing top categories per user"):
        category_counts = user_category_counts[user_id]
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        top_5_per_user[user_id] = sorted_categories[:5]
    
    return top_5_per_user


def compute_topics(cfg):
    graph, _= load_data(**cfg.dataset.download, cfg=cfg)
    
    # Usage
    top_5_categories = get_top_5_news_per_user(graph)
        
        
    df_data = {}
    for user_id, categories in top_5_categories.items():
        df_data[user_id] = [cat for cat, _ in categories] + [float('nan')] * (5 - len(categories))

    df = pd.DataFrame.from_dict(df_data, orient='index', columns=['Category1', 'Category2', 'Category3', 'Category4', 'Category5'])
    
    file_path = os.path.join(cfg.dataset_dir, 'user_topics.parquet')
    df.to_parquet(file_path)


def eval_user_topics(cfg):
    graph, dataset = load_data(**cfg.dataset.download, cfg=cfg)
    all_auc = 0
    all_mrr = 0
    all_ndcg_5 = 0
    all_ndcg_10 = 0
    
    all_TP_ratio = 0
    all_FP_ratio = 0
    all_TN_ratio = 0
    all_FN_ratio = 0
    
    df = pd.read_parquet(os.path.join(cfg.dataset_dir, '.user_topics.parquet'))
    user_top_5_categories = {}
    for user_id, row in df.iterrows():
        user_top_5_categories[int(user_id)] = [cat for cat in row if not pd.isna(cat)]
        
    news_subcategories = graph.ndata['SubCategory']['news']
    summed_categories = torch.sum(news_subcategories, dim=1)

    dev_session_loader = dataset.get_dev_session_loader(shuffle=False)
    devloader = tqdm(enumerate(dev_session_loader))
    not_found = 0
    num_users = 0
    
    for i, (pos_links, neg_links) in devloader:
        no_history = False
        pos_edges = graph.find_edges(pos_links, etype=('news', 'pos_dev_r', 'user'))
        pos_news_ids, pos_user_ids = pos_edges

        # Retrieve edge data for negative links
        neg_edges = graph.find_edges(neg_links, etype=('news', 'neg_dev_r', 'user'))
        neg_news_ids, neg_user_ids = neg_edges
        
        all_news_ids = torch.cat([pos_news_ids, neg_news_ids])
        all_user_ids = torch.cat([pos_user_ids, neg_user_ids])
        
        scores = []
        labels = []
        for news_id, user_id in zip(all_news_ids, all_user_ids):
            news_id = news_id.item()
            user_id = user_id.item()
            if user_id in user_top_5_categories.keys() and len(user_top_5_categories[user_id]) == 5 and news_id in range(len(summed_categories)):
                user_top_5 = user_top_5_categories[user_id]
                news_subcategory = summed_categories[news_id]
                score = 1 if news_subcategory in user_top_5 else 0
            else:
                print(f"User or news not found ({not_found})")
                not_found += 1
                no_history = True
                break
            scores.append(score)

        if no_history:
            continue
        num_users += 1
        
        scores = torch.tensor(scores)
        labels = torch.cat([torch.ones(len(pos_links)), torch.zeros(len(neg_links))])
        
        # Compute metrics
        accuracy = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)
               
        TP = np.sum((scores.numpy() == 1) & (labels.numpy() == 1))
        FP = np.sum((scores.numpy() == 1) & (labels.numpy() == 0))
        TN = np.sum((scores.numpy() == 0) & (labels.numpy() == 0))
        FN = np.sum((scores.numpy() == 0) & (labels.numpy() == 1))

        # Compute ratios
        total_positives = TP + FN
        total_negatives = TN + FP

        all_TP_ratio += TP / total_positives if total_positives > 0 else 0
        all_FP_ratio +=  FP / total_negatives if total_negatives > 0 else 0
        all_TN_ratio += TN / total_negatives if total_negatives > 0 else 0
        all_FN_ratio +=  FN / total_positives if total_positives > 0 else 0
                
        all_auc += accuracy
        all_mrr += mrr_score
        all_ndcg_5 += ndcg_5
        all_ndcg_10 += ndcg_10

        
    results = {
        'accuracy': all_auc / num_users,
        'mrr': all_mrr / num_users,
        'ndcg@5': all_ndcg_5 / num_users,
        'ndcg@10': all_ndcg_10 / num_users,
        'TP_ratio': all_TP_ratio / num_users,
        'FP_ratio': all_FP_ratio / num_users,
        'TN_ratio': all_TN_ratio / num_users,
        'FN_ratio': all_FN_ratio / num_users
    }
    print(results)
        
    return results
