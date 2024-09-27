
import pandas as pd
import numpy as np
import torch
import time
import json
from tqdm import tqdm
import time
import pickle
from pathlib import Path
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from data.utils.node2vec.model import Node2vecModel
from data.utils.node2vec.utils import load_graph, parse_arguments



def strid_intid(str_id):
    return int(str_id[1:])


def seq_list(l):
    num_lines = len(l)
    lines = []
    for i in range(num_lines):
        ll = [str(j) for j in l[i]]
        lines.append('\t'.join(ll))
    return '\n'.join(lines)


def unseq_list(seq):
    l = []
    lines = seq.split('\n')
    for line in lines:
        line_split = line.split('\t')
        line_split = [int(id_str) for id_str in line_split]
        l.append(line_split)
    return l


def seq_numpy(nparray):
    line = []
    for i in range(nparray.shape[0]):
        line.append('\t'.join(str(n) for n in nparray[i]))
    return '\n'.join(str(m) for m in line)


def unseq_numpy(seq):
    lines = seq.split('\n')
    ar = []
    for line in lines:
        ar.append([int(i) for i in line.split('\t')])
    return np.array(ar)


def save_list(l, path):
    with open(path, 'w') as f:
        f.write(seq_list(l))


def read_list(path):
    with open(path, 'r') as f:
        return unseq_list(f.read())


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def tokenize_sentence(sentence, stop_words):
    words = word_tokenize(sentence)
    words_filtered = []
    for w in words:
        if w not in stop_words:
            words_filtered.append(w)
    return words


def parse_time(time_str):
    t = time.strptime(time_str, "%m/%d/%Y %I:%M:%S %p")
    return (t.tm_yday - 313) * 24 * 60 * 60 + t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec


def parse_entity_record(entities):
    entities_list = []
    try:
        entities = entities[2:-2]
        entities_split = entities.split('}, {')
        for entity in entities_split:
            entity_json = json.loads('{' + entity + '}')
            entities_list.append(entity_json['WikidataId'])
    except Exception as e:
        pass
    return entities_list


def tokenize_news(train_news_df, dev_news_df, newsid2nid, save_dir):
    print('tokenize_news')
    nw_link = []  # [nid, wid]
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    train_keywords = np.concatenate([str(kws).split(' ') for kws in train_news_df['Keywords'].values])
    dev_keywords = np.concatenate([str(kws).split(' ') for kws in dev_news_df['Keywords'].values])
    words = np.unique(np.concatenate([train_keywords, dev_keywords]))
    word_emb = sentence_model.encode(words)
    word2wid = {str(v): k for k, v in enumerate(words)}  # word -> wid
    for news_df in [train_news_df, dev_news_df]:
        for i in tqdm(range(news_df.shape[0])):
            nid = newsid2nid[news_df.index[i]]
            news_keywords = str(news_df.iloc[i]['Keywords']).split(' ')
            for kwd in news_keywords:
                nw_link.append([nid, word2wid[kwd]])
    return np.unique(np.array(nw_link), axis=0), words, word_emb


def get_news_entity_link(news_df, newsid2nid, entity_embedding_dict, save_dir, force_reload=False):
    print('get_news_entity_link')
    try:
        if force_reload:
            raise Exception('Force Reload.')
        news_entity_link = np.load('{}/news_entity_link.npy'.format(save_dir))
        print('load from files')
    except Exception as e:
        rows_num = news_df.shape[0]
        news_entity_link = []  # pair (nid, WikidataIds)
        no_entity_count = 0
        for i in tqdm(range(rows_num)):
            nid = newsid2nid[news_df.index[i]]
            title_entities = news_df.iloc[i]['Title-Entities']
            abstract_entities = news_df.iloc[i]['Abstract-Entities']
            title_entities_list = parse_entity_record(title_entities)  # [WikidataIds]
            abstract_entities_list = parse_entity_record(abstract_entities)  # [WikidataIds]
            entities_list = np.unique(np.concatenate([np.array(title_entities_list), np.array(abstract_entities_list)]))
            entities_list = [ett_id for ett_id in entities_list if ett_id in entity_embedding_dict]  # filtering entities without embedding
            if len(entities_list) > 0:
                for ett_id in entities_list:
                    news_entity_link.append([nid, strid_intid(ett_id)])  
            else:
                no_entity_count += 1
        print('>>> News w/o entity: {}/{}'.format(no_entity_count, rows_num))
        news_entity_link = np.array(news_entity_link)
        np.save('{}/news_entity_link.npy'.format(save_dir), news_entity_link)
    return news_entity_link


def get_news_list(train_behaviors_df, dev_behaviors_df):
    df = [train_behaviors_df, dev_behaviors_df]
    news_list = [[], []]
    for s in range(2):
        rows_num = df[s].shape[0]
        for i in tqdm(range(rows_num)):
            history = df[s].iloc[i]['History']
            if not isinstance(history, float):
                for nid in history.split():
                    news_list[s].append(nid)
            impressions = df[s].iloc[i]['Impressions']
            if not isinstance(impressions, float):
                for nid in impressions.split():
                    news_list[s].append(nid[:-2])  # remove labels
    return np.unique(np.array(news_list[0]+news_list[1]))


def get_user_news_link(behaviors_df, userid2uid, newsid2nid, save_dir, force_reload=False):
    print('get_user_news_link')
    try:
        if force_reload:
            raise Exception('Force Reload.')
        history_actions = np.load('{}/history_actions.npy'.format(save_dir))
        positive_actions = np.load('{}/positive_actions.npy'.format(save_dir))
        negative_actions = np.load('{}/negative_actions.npy'.format(save_dir))
        session_positive = read_list('{}/session_positive.txt'.format(save_dir))
        session_negative = read_list('{}/session_negative.txt'.format(save_dir))
        print('load from files')
    except Exception as e:
        rows_num = behaviors_df.shape[0]
        history_actions = []  # pair (uid, nid)
        positive_actions = []  # triple (uid, nid, t)
        negative_actions = []  # triple (uid, nid, t)
        positive_id = 0  # global count for assigning EIDs
        negative_id = 0  # global count for assigning EIDs
        session_positive = []  # [[EIDs of positive impressions]]
        session_negative = []  # [[EIDs of negative impressions]]

        for i in tqdm(range(rows_num)):  # processing per-record
            
            uid = userid2uid[behaviors_df.iloc[i]['User-ID']]
            pos_link = []  # positive link EIDs of this user
            neg_link = []  # negative link EIDs of this user

            t = int(behaviors_df.iloc[i]['Time'])

            history = behaviors_df.iloc[i]['History']
            if not isinstance(history, float):
                for newsid in history.split():
                    history_actions.append([uid, newsid2nid[newsid]])

            impressions = behaviors_df.iloc[i]['Impressions']
            if not isinstance(impressions, float):
                for newsid in impressions.split():
                    state = newsid[-1]
                    newsid = newsid[:-2]
                    if state == '1':
                        positive_actions.append([uid, newsid2nid[newsid], t])  # node-node for make graph
                        pos_link.append(positive_id)  # edge id for call it out
                        positive_id += 1
                    else:
                        negative_actions.append([uid, newsid2nid[newsid], t])
                        neg_link.append(negative_id)
                        negative_id += 1
            
            session_positive.append(np.array(pos_link))
            session_negative.append(np.array(neg_link))

        history_actions = np.array(history_actions)
        positive_actions = np.array(positive_actions)
        negative_actions = np.array(negative_actions)
        np.save('{}/history_actions.npy'.format(save_dir), history_actions)
        np.save('{}/positive_actions.npy'.format(save_dir), positive_actions)
        np.save('{}/negative_actions.npy'.format(save_dir), negative_actions)
        save_list(session_positive, '{}/session_positive.txt'.format(save_dir))
        save_list(session_negative, '{}/session_negative.txt'.format(save_dir))

    return history_actions, positive_actions, negative_actions, session_positive, session_negative


def building_training_dataset(session_positive, session_negative, gnn_neg_ratio):
    sampled_datasets = []  # training edge dataset
    for i in range(len(session_positive)):
        pos_link = np.array(session_positive[i])
        neg_link = np.array(session_negative[i])
        len_pos = len(pos_link)
        len_neg = len(neg_link)
        # negative sampling to build training datasets
        for j in range(len_pos):
            if len_neg < gnn_neg_ratio:
                nega_sample = []
                for s in range(int(gnn_neg_ratio / len_neg)):
                    nega_sample.append(neg_link)
                neg_sample_id = np.random.choice(len_neg, gnn_neg_ratio % len_neg, replace=False)
                nega_sample.append(neg_link[neg_sample_id])
                nega_sample = np.concatenate(nega_sample)
            else:
                neg_sample_id = np.random.choice(len_neg, gnn_neg_ratio, replace=False)
                nega_sample = neg_link[neg_sample_id]
            sampled_datasets.append(np.concatenate([np.array([pos_link[j]]), nega_sample]))
    return np.array(sampled_datasets)


def load_MIND(save_dir, force_reload=False):  # New Version for v7
    try:
        if force_reload:
            raise Exception('Force Reload.')
        news_file_name = 'preprocessed_news.csv'
        news_file_path = '{}/{}'.format(save_dir, news_file_name)
        news_df = pd.read_csv(news_file_path, sep='\t', index_col='News-ID')
        news_df['Title-Emb'] = [np.array([float(v) for v in emb_text.replace('\n', '')[1:-1].split()]) for emb_text in news_df['Title-Emb']]
        news_df['Abstract-Emb'] = [np.array([float(v) for v in emb_text.replace('\n', '')[1:-1].split()]) for emb_text in news_df['Abstract-Emb']]
        print('load from files')
    except Exception as e:
        news_file_name = 'news.tsv'
        news_file_path = '{}/{}'.format(save_dir, news_file_name)
        news_df = pd.read_csv(news_file_path, sep='\t', names=["News-ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title-Entities", "Abstract-Entities"], index_col='News-ID')
        news_df['row_index'] = range(news_df.shape[0])

        # encoding and keyword extracting
        sentence_model = SentenceTransformer("all-mpnet-base-v2")
        kw_model = KeyBERT(model=sentence_model)

        title_embedding = sentence_model.encode([str(t) for t in news_df['Title'].values])
        abstract_embedding = sentence_model.encode([str(t) for t in news_df['Abstract'].values])
        full_text = []
        for i in range(news_df['Title'].values.shape[0]):
            full_text.append(str(news_df['Title'].values[i]) + '. ' + str(news_df['Abstract'].values[i]))
        keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 1), stop_words='english')
        keywords = [' '.join(w[0] for w in kw) for kw in keywords]

        news_df['Title-Emb'] = news_df.apply(lambda row: title_embedding[row.row_index], axis=1)
        news_df['Abstract-Emb'] = news_df.apply(lambda row: abstract_embedding[row.row_index], axis=1)
        news_df['Keywords'] = keywords
        news_df.to_csv('{}/preprocessed_news.csv'.format(save_dir), sep='\t')

    # Load behaviors.tsv
    behaviors_file_name = 'behaviors.tsv'
    behaviors_file_path = '{}/{}'.format(save_dir, behaviors_file_name)
    behaviors_df = pd.read_csv(behaviors_file_path, sep='\t', names=["Impression-ID", "User-ID", "Time", "History", "Impressions"], index_col='Impression-ID')
    # Reformat timestamp
    behaviors_df['Time'] = behaviors_df['Time'].apply(parse_time)
    behaviors_df = behaviors_df.sort_values(by=['Time'], na_position='first')

    # Load entity_embedding.vec
    entity_embedding_file_name = 'entity_embedding.vec'
    entity_embedding_file_path = '{}/{}'.format(save_dir, entity_embedding_file_name)
    entity_embedding_dict = {}
    with open(entity_embedding_file_path, 'r') as f:
        for line in f:
            line_split = line.split('\t')[:-1]
            entity_id = line_split[0]
            entity_emb = np.array([float(i) for i in line_split[1:]])
            entity_embedding_dict[entity_id] = entity_emb

    return behaviors_df, news_df, entity_embedding_dict


def node2vec(mind_dgl, save_dir, force_reload=False):
    try:
        if force_reload:
            raise Exception('Force Reload.')
        node2vec = torch.load('{}/MINDsmall/node2vec.pth'.format(save_dir))
        print('load from files')
    except Exception as e:
        args = parse_arguments()
        graph = load_graph(mind_dgl)
        trainer = Node2vecModel(graph,
                    embedding_dim=args.embedding_dim,
                    walk_length=args.walk_length,
                    p=args.p,
                    q=args.q,
                    num_walks=args.num_walks,
                    device=args.device)
        trainer.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=1e-2)
        node2vec = torch.load('{}/MINDsmall/node2vec.pth'.format(save_dir))
    # entity2vec = node2vec[:mind_dgl.num_node['entity']]
    # node2vec = node2vec[mind_dgl.num_node['entity']:]
    news2vec = node2vec[:mind_dgl.num_node['news']]
    node2vec = node2vec[mind_dgl.num_node['news']:]
    user2vec = node2vec[:mind_dgl.num_node['user']]
    node2vec = node2vec[mind_dgl.num_node['user']:]
    # word2vec = node2vec[:mind_dgl.num_node['word']]
    # node2vec = node2vec[mind_dgl.num_node['word']:]
    return news2vec, user2vec
    # return entity2vec, news2vec, user2vec, word2vec
