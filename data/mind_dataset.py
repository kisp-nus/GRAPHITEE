import random
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader
import os
import time
import json
import numpy as np
import dgl
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info
import pandas as pd
import s3fs
import glob
import pickle
from sentence_transformers import SentenceTransformer

from data.utils.mind_utils import seq_list, unseq_list
from data.utils.mind_utils import load_MIND, get_news_list, get_user_news_link, get_news_entity_link, tokenize_news, building_training_dataset, node2vec


class MIND_DGL(DGLDataset):
    """
    MIND DGL Graph Constructor v7
    
    Credits: DivHGNN's implementation by @GuangpingZhang (Github).
    """

    def __init__(
        self,
        cfg: DictConfig,
        name='MIND_DGLHIN_v7',
        url=None,
        hash_key='MIND DGL Graph Constructor v7',
        force_reload=False,
        verbose=True
    ):
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._raw_dir = cfg["dataset_dir"]
        self._save_dir = cfg["dataset_dir"]
        self._indices = None
        
        self.cfg = cfg

        self._graph = None

        self._num_node = None
        self._num_link = None
        self._reverse_etypes = None

        self._train_session_positive = None
        self._train_session_negative = None
        self._dev_session_positive = None
        self._dev_session_negative = None

        self._user = None  # User List uid -> user_id
        self._news = None  # News List nid -> news_id
        self._word = None  # Word list wid -> word
        self._entity = None

        self._graph_path = self._save_dir + \
            '/graph_{}.json'.format(cfg['mind_version'])
        self._save_info = self._save_dir + \
            '/info_{}.json'.format(cfg['mind_version'])

        self._load()

    def process(self):
        r"""
        Process the raw data to graph, without word and entity nodes. 
        """
        self._print('Loading raw data ...')
        train_behaviors_df, train_news_df, train_entity_embedding_dict = \
            load_MIND('{}/MIND{}_train'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), force_reload=False)#self._force_reload)
        dev_behaviors_df, dev_news_df, dev_entity_embedding_dict = \
            load_MIND('{}/MIND{}_dev'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), force_reload=False)#self._force_reload)

        self._print('Making User-List and News-List ...')
        # User List and News List
        self._user = np.unique(np.array(list(train_behaviors_df['User-ID']) + list(dev_behaviors_df['User-ID'])))  # [User-ID]
        userid2uid = {v: k for k, v in enumerate(self._user)}  # uid: the NID of user nodes in the graph
        self._news = get_news_list(train_behaviors_df, dev_behaviors_df)  # [News-ID]
        newsid2nid = {v: k for k, v in enumerate(self._news)}  # nid: the NID of news nodes in the graph

        self._print('Extracting User-News ...')
        # get u-n
        # For graph: history, train_pos, train_neg, dev_pos, dev_neg
        # For dataloader: session, sample
        train_history_actions, train_positive_actions, train_negative_actions, train_session_positive, train_session_negative = \
            get_user_news_link(train_behaviors_df, userid2uid, newsid2nid, '{}/MIND{}_train'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        dev_history_actions, dev_positive_actions, dev_negative_actions, dev_session_positive, dev_session_negative = \
            get_user_news_link(dev_behaviors_df, userid2uid, newsid2nid, '{}/MIND{}_dev'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        history_actions = np.unique(np.concatenate([train_history_actions, dev_history_actions]), axis=0)
        # session & sample -> save to self for constucting dataloader
        self._train_session_positive = train_session_positive
        self._train_session_negative = train_session_negative
        self._dev_session_positive = dev_session_positive
        self._dev_session_negative = dev_session_negative

        # self._print('Extracting News-Entity ...')
        # processing enid and get n-e
        entity_embedding_dict = {**train_entity_embedding_dict, **dev_entity_embedding_dict}
        train_news_entity_link = get_news_entity_link(train_news_df, newsid2nid, entity_embedding_dict, '{}/MIND{}_train'.format(
            self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        dev_news_entity_link = get_news_entity_link(dev_news_df, newsid2nid, entity_embedding_dict, '{}/MIND{}_dev'.format(
            self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        news_entity_link = np.unique(np.concatenate([train_news_entity_link, dev_news_entity_link]), axis=0)
        self._entity = [int(v) for v in np.unique(news_entity_link[:, 1]).tolist()]
        entityid2enid = {v: k for k, v in enumerate(self._entity)}  # WikidataIds -> enid
        news_entity_link[:, 1] = [entityid2enid[oid] for oid in news_entity_link[:, 1]]

        # self._print('Extracting News-Word ...')
        # # get n-w and emb_w(wid -> emb)
        # news_word_link, words, word_emb = tokenize_news(train_news_df, dev_news_df, newsid2nid, self.cfg['dataset_dir'])
        # self._word = words

        self._print('Building Nodes&Edges List ...')
        _nodes = {
            'user': torch.Tensor(list(range(len(self._user)))).type(torch.int32),
            'news': torch.Tensor(list(range(len(self._news)))).type(torch.int32),
            'entity': torch.Tensor(list(range(len(self._entity)))).type(torch.int32),
            # 'word': torch.Tensor(list(range(len(self._word)))).type(torch.int32),
        }

        self._num_node = {}   # need save
        for node_type in _nodes:
            self._num_node[node_type] = int(_nodes[node_type].max()) + 1

        un_df = pd.DataFrame(history_actions, columns=['uid', 'nid'])
        ne_df = pd.DataFrame(news_entity_link, columns=['nid', 'enid'])
        # nw_df = pd.DataFrame(news_word_link, columns=['nid', 'wid'])
        ue_df = un_df.join(ne_df.set_index('nid'), on='nid', how='inner').sort_values(by=['uid'])[['uid', 'enid']]
        # uw_df = un_df.join(nw_df.set_index('nid'), on='nid', how='inner').sort_values(by=['uid'])[['uid', 'wid']]

        _links = {
            ('user', 'history', 'news'): np.array(history_actions).tolist(),
            ('user', 'pos_train', 'news'): np.array(train_positive_actions)[:, :2].tolist(),
            ('user', 'pos_dev', 'news'): np.array(dev_positive_actions)[:, :2].tolist(),
            ('user', 'neg_train', 'news'): np.array(train_negative_actions)[:, :2].tolist(),
            ('user', 'neg_dev', 'news'): np.array(dev_negative_actions)[:, :2].tolist(),
            ('news', 'ne_link', 'entity'): np.array(news_entity_link).tolist(),
            # ('news', 'nw_link', 'word'): np.array(news_word_link).tolist(),
            ('user', 'ue_link', 'entity'): np.array(ue_df).tolist(),
            # ('user', 'uw_link', 'word'): np.array(uw_df).tolist(),
        }  # in graph, no saving need

        # build reverse links
        self._reverse_etypes = {}   # need save
        cur_relations = list(_links.keys())
        for relation in cur_relations:
            reverse_relation = '{}_r'.format(relation[1])
            _links[(relation[2], reverse_relation, relation[0])] = [[p[1], p[0]] for p in _links[relation]]
            self._reverse_etypes[relation[1]] = reverse_relation
            self._reverse_etypes[reverse_relation] = relation[1]

        # add self loop
        for node_type in self.num_node:
            node_ids = list(range(self.num_node[node_type]))
            self_link = np.array([node_ids, node_ids]).T.tolist()
            _links[(node_type, '{}_selfloop'.format(
                node_type), node_type)] = self_link

        self._num_link = {}   # need save
        for link_type in _links:
            self._num_link[link_type[1]] = len(_links[link_type])

        self._print('Building Graph ...')
        self._graph = dgl.heterograph(_links, num_nodes_dict=self._num_node, idtype=torch.int32)  # need save

        self._print('Processing Nodes Atrrs ...')
        _embedding = {node_type: {} for node_type in _nodes}

        self._print('Processing News')
        _categories = pd.concat([train_news_df, dev_news_df], axis=0)['Category'].unique()
        _news_category_map = {name: i for i, name in enumerate(_categories)}  # category -> cateid
        _subcategories = pd.concat([train_news_df, dev_news_df], axis=0)['SubCategory'].unique()
        _news_subcategory_map = {name: i for i, name in enumerate(_subcategories)}  # subcategory -> subcateid

        _embedding['news']['News_Title_Embedding'] = torch.zeros([self._num_node['news'], 384])
        _embedding['news']['News_Abstract_Embedding'] = torch.zeros([self._num_node['news'], 384])
        _news_category = torch.zeros([self._num_node['news']])
        _news_subcategory = torch.zeros([self._num_node['news']])
        _empty_news_emb = 0
        for nid, news_id in enumerate(self._news):
            if news_id in train_news_df.index:
                _embedding['news']['News_Title_Embedding'][nid] = torch.Tensor(train_news_df['Title-Emb'][news_id])
                _embedding['news']['News_Abstract_Embedding'][nid] = torch.Tensor(train_news_df['Abstract-Emb'][news_id])
                _news_category[nid] = _news_category_map[train_news_df['Category'][news_id]]
                _news_subcategory[nid] = _news_subcategory_map[train_news_df['SubCategory'][news_id]]
            elif news_id in dev_news_df.index:
                _embedding['news']['News_Title_Embedding'][nid] = torch.Tensor(dev_news_df['Title-Emb'][news_id])
                _embedding['news']['News_Abstract_Embedding'][nid] = torch.Tensor(dev_news_df['Abstract-Emb'][news_id])
                _news_category[nid] = _news_category_map[dev_news_df['Category'][news_id]]
                _news_subcategory[nid] = _news_subcategory_map[dev_news_df['SubCategory'][news_id]]
            if _embedding['news']['News_Title_Embedding'][nid].sum() == 0 or _embedding['news']['News_Abstract_Embedding'][nid].sum() == 0:
                _empty_news_emb += 1
        _embedding['news']['CateID'] = _news_category.long()  # cateid
        _embedding['news']['SubCateID'] = _news_subcategory.long()  # subcateid
        sentence_model = SentenceTransformer("all-mpnet-base-v2")#all-MiniLM-L6-v2
        category_embeddings = sentence_model.encode(_categories)
        subcategory_embeddings = sentence_model.encode(_subcategories)
        _embedding['news']['Category'] = torch.mm(torch.nn.functional.one_hot(_news_category.long()).float(), torch.Tensor(category_embeddings))
        _embedding['news']['SubCategory'] = torch.mm(torch.nn.functional.one_hot(_news_subcategory.long()).float(), torch.Tensor(subcategory_embeddings))
        print('Num of Empty News Embedding: {} in {}'.format(_empty_news_emb, len(_nodes['news'])))

        # self._print('Processing Entity')
        # new entity id -> emb
        _embedding['entity']['Entity_Embedding'] = torch.zeros([self._num_node['entity'], 100])  # 100 as entity embedding dim
        for id, oid in enumerate(self._entity):  # WikidataID
            entity_wiki_id = 'Q{}'.format(oid)
            _embedding['entity']['Entity_Embedding'][id] = torch.Tensor(entity_embedding_dict[entity_wiki_id])

        # self._print('Processing Word')
        # _embedding['word']['Word_Embedding'] = torch.Tensor(word_emb)

        self._print('Processing User')
        _embedding['user']['Category'] = torch.ones([self._num_node['user'], len(_categories)]) * 1e-10  # avoid zero
        _embedding['user']['SubCategory'] = torch.ones([self._num_node['user'], len(_subcategories)]) * 1e-10
        for i in range(history_actions.shape[0]):
            uid = history_actions[i][0]
            nid = history_actions[i][1]
            news_id = self._news[nid]
            if news_id in train_news_df.index:
                cid = _news_category_map[train_news_df['Category'][news_id]]
                scid = _news_subcategory_map[train_news_df['SubCategory'][news_id]]
                _embedding['user']['Category'][uid, cid] += 1
                _embedding['user']['SubCategory'][uid, scid] += 1
            elif news_id in dev_news_df.index:
                cid = _news_category_map[dev_news_df['Category'][news_id]]
                scid = _news_subcategory_map[dev_news_df['SubCategory'][news_id]]
                _embedding['user']['Category'][uid, cid] += 1
                _embedding['user']['SubCategory'][uid, scid] += 1
        _embedding['user']['Category'] = torch.mm(_embedding['user']['Category'], torch.Tensor(category_embeddings))
        _embedding['user']['SubCategory'] = torch.mm(_embedding['user']['SubCategory'], torch.Tensor(subcategory_embeddings))

        # Mounting
        for ntype in _embedding:
            for emb_type in _embedding[ntype]:
                self._graph.nodes[ntype].data[emb_type] = _embedding[ntype][emb_type]
        
        # # node2vec
        # self._print("Perform training node2vec model")
        # news2vec, user2vec = node2vec(self, self.cfg['dataset_dir'], force_reload=self._force_reload) 
        
        # # self.graph.nodes['entity'].data['Node2Vec'] = entity2vec
        # self.graph.nodes['news'].data['Node2Vec'] = news2vec
        # self.graph.nodes['user'].data['Node2Vec'] = user2vec
        # # self.graph.nodes['word'].data['Node2Vec'] = word2vec

        # Edge Attr -- Time
        self._print('Time')
        self._graph.edges['pos_train'].data['Time'] = torch.Tensor(
            train_positive_actions.T[2]).type(torch.float)
        self._graph.edges['pos_dev'].data['Time'] = torch.Tensor(
            dev_positive_actions.T[2]).type(torch.float)
        self._graph.edges['neg_train'].data['Time'] = torch.Tensor(
            train_negative_actions.T[2]).type(torch.float)
        self._graph.edges['neg_dev'].data['Time'] = torch.Tensor(
            dev_negative_actions.T[2]).type(torch.float)
        self._graph.edges['pos_train_r'].data['Time'] = torch.Tensor(
            train_positive_actions.T[2]).type(torch.float)
        self._graph.edges['pos_dev_r'].data['Time'] = torch.Tensor(
            dev_positive_actions.T[2]).type(torch.float)
        self._graph.edges['neg_train_r'].data['Time'] = torch.Tensor(
            train_negative_actions.T[2]).type(torch.float)
        self._graph.edges['neg_dev_r'].data['Time'] = torch.Tensor(
            dev_negative_actions.T[2]).type(torch.float)

        # Edge Attr -- Label
        self._print('Label')
        self._graph.edges['pos_train'].data['Label'] = torch.ones(
            self._graph.edges['pos_train'].data['Time'].shape)
        self._graph.edges['pos_dev'].data['Label'] = torch.ones(
            self._graph.edges['pos_dev'].data['Time'].shape)
        self._graph.edges['neg_train'].data['Label'] = torch.zeros(
            self._graph.edges['neg_train'].data['Time'].shape)
        self._graph.edges['neg_dev'].data['Label'] = torch.zeros(
            self._graph.edges['neg_dev'].data['Time'].shape)
        self._graph.edges['pos_train_r'].data['Label'] = torch.ones(
            self._graph.edges['pos_train_r'].data['Time'].shape)
        self._graph.edges['pos_dev_r'].data['Label'] = torch.ones(
            self._graph.edges['pos_dev_r'].data['Time'].shape)
        self._graph.edges['neg_train_r'].data['Label'] = torch.zeros(
            self._graph.edges['neg_train_r'].data['Time'].shape)
        self._graph.edges['neg_dev_r'].data['Label'] = torch.zeros(
            self._graph.edges['neg_dev_r'].data['Time'].shape)
    
    
    def process_hetero_div(self):
        r"""
        initial process method from the paper. 
        """
        self._print('Loading raw data ...')
        train_behaviors_df, train_news_df, train_entity_embedding_dict = \
            load_MIND('{}/MIND{}_train'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        dev_behaviors_df, dev_news_df, dev_entity_embedding_dict = \
            load_MIND('{}/MIND{}_dev'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)

        self._print('Making User-List and News-List ...')
        # User List and News List
        self._user = np.unique(np.array(list(train_behaviors_df['User-ID']) + list(dev_behaviors_df['User-ID'])))  # [User-ID]
        userid2uid = {v: k for k, v in enumerate(self._user)}  # uid: the NID of user nodes in the graph
        self._news = get_news_list(train_behaviors_df, dev_behaviors_df)  # [News-ID]
        newsid2nid = {v: k for k, v in enumerate(self._news)}  # nid: the NID of news nodes in the graph

        self._print('Extracting User-News ...')
        # get u-n
        # For graph: history, train_pos, train_neg, dev_pos, dev_neg
        # For dataloader: session, sample
        train_history_actions, train_positive_actions, train_negative_actions, train_session_positive, train_session_negative = \
            get_user_news_link(train_behaviors_df, userid2uid, newsid2nid, '{}/MIND{}_train'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        dev_history_actions, dev_positive_actions, dev_negative_actions, dev_session_positive, dev_session_negative = \
            get_user_news_link(dev_behaviors_df, userid2uid, newsid2nid, '{}/MIND{}_dev'.format(self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        history_actions = np.unique(np.concatenate([train_history_actions, dev_history_actions]), axis=0)
        # session & sample -> save to self for constucting dataloader
        self._train_session_positive = train_session_positive
        self._train_session_negative = train_session_negative
        self._dev_session_positive = dev_session_positive
        self._dev_session_negative = dev_session_negative

        self._print('Extracting News-Entity ...')
        # processing enid and get n-e
        entity_embedding_dict = {**train_entity_embedding_dict, **dev_entity_embedding_dict}
        train_news_entity_link = get_news_entity_link(train_news_df, newsid2nid, entity_embedding_dict, '{}/MIND{}_train'.format(
            self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        dev_news_entity_link = get_news_entity_link(dev_news_df, newsid2nid, entity_embedding_dict, '{}/MIND{}_dev'.format(
            self.cfg['dataset_dir'], self.cfg['mind_version']), self._force_reload)
        news_entity_link = np.unique(np.concatenate([train_news_entity_link, dev_news_entity_link]), axis=0)
        self._entity = [int(v) for v in np.unique(news_entity_link[:, 1]).tolist()]
        entityid2enid = {v: k for k, v in enumerate(self._entity)}  # WikidataIds -> enid
        news_entity_link[:, 1] = [entityid2enid[oid] for oid in news_entity_link[:, 1]]

        self._print('Extracting News-Word ...')
        # get n-w and emb_w(wid -> emb)
        news_word_link, words, word_emb = tokenize_news(train_news_df, dev_news_df, newsid2nid, self.cfg['dataset_dir'])
        self._word = words

        self._print('Building Nodes&Edges List ...')
        _nodes = {
            'user': torch.Tensor(list(range(len(self._user)))).type(torch.int32),
            'news': torch.Tensor(list(range(len(self._news)))).type(torch.int32),
            'entity': torch.Tensor(list(range(len(self._entity)))).type(torch.int32),
            'word': torch.Tensor(list(range(len(self._word)))).type(torch.int32),
        }

        self._num_node = {}   # need save
        for node_type in _nodes:
            self._num_node[node_type] = int(_nodes[node_type].max()) + 1

        un_df = pd.DataFrame(history_actions, columns=['uid', 'nid'])
        ne_df = pd.DataFrame(news_entity_link, columns=['nid', 'enid'])
        nw_df = pd.DataFrame(news_word_link, columns=['nid', 'wid'])
        ue_df = un_df.join(ne_df.set_index('nid'), on='nid', how='inner').sort_values(by=['uid'])[['uid', 'enid']]
        uw_df = un_df.join(nw_df.set_index('nid'), on='nid', how='inner').sort_values(by=['uid'])[['uid', 'wid']]

        _links = {
            ('user', 'history', 'news'): np.array(history_actions).tolist(),
            ('user', 'pos_train', 'news'): np.array(train_positive_actions)[:, :2].tolist(),
            ('user', 'pos_dev', 'news'): np.array(dev_positive_actions)[:, :2].tolist(),
            ('user', 'neg_train', 'news'): np.array(train_negative_actions)[:, :2].tolist(),
            ('user', 'neg_dev', 'news'): np.array(dev_negative_actions)[:, :2].tolist(),
            ('news', 'ne_link', 'entity'): np.array(news_entity_link).tolist(),
            ('news', 'nw_link', 'word'): np.array(news_word_link).tolist(),
            ('user', 'ue_link', 'entity'): np.array(ue_df).tolist(),
            ('user', 'uw_link', 'word'): np.array(uw_df).tolist(),
        }  # in graph, no saving need

        # build reverse links
        self._reverse_etypes = {}   # need save
        cur_relations = list(_links.keys())
        for relation in cur_relations:
            reverse_relation = '{}_r'.format(relation[1])
            _links[(relation[2], reverse_relation, relation[0])] = [[p[1], p[0]] for p in _links[relation]]
            self._reverse_etypes[relation[1]] = reverse_relation
            self._reverse_etypes[reverse_relation] = relation[1]

        # add self loop
        for node_type in self.num_node:
            node_ids = list(range(self.num_node[node_type]))
            self_link = np.array([node_ids, node_ids]).T.tolist()
            _links[(node_type, '{}_selfloop'.format(
                node_type), node_type)] = self_link

        self._num_link = {}   # need save
        for link_type in _links:
            self._num_link[link_type[1]] = len(_links[link_type])

        self._print('Building Graph ...')
        self._graph = dgl.heterograph(_links, num_nodes_dict=self._num_node, idtype=torch.int32)  # need save

        self._print('Processing Nodes Atrrs ...')
        _embedding = {node_type: {} for node_type in _nodes}

        self._print('Processing News')
        _categories = pd.concat([train_news_df, dev_news_df], axis=0)['Category'].unique()
        _news_category_map = {name: i for i, name in enumerate(_categories)}  # category -> cateid
        _subcategories = pd.concat([train_news_df, dev_news_df], axis=0)['SubCategory'].unique()
        _news_subcategory_map = {name: i for i, name in enumerate(_subcategories)}  # subcategory -> subcateid

        _embedding['news']['News_Title_Embedding'] = torch.zeros([self._num_node['news'], 384])
        _embedding['news']['News_Abstract_Embedding'] = torch.zeros([self._num_node['news'], 384])
        _news_category = torch.zeros([self._num_node['news']])
        _news_subcategory = torch.zeros([self._num_node['news']])
        _empty_news_emb = 0
        for nid, news_id in enumerate(self._news):
            if news_id in train_news_df.index:
                _embedding['news']['News_Title_Embedding'][nid] = torch.Tensor(train_news_df['Title-Emb'][news_id])
                _embedding['news']['News_Abstract_Embedding'][nid] = torch.Tensor(train_news_df['Abstract-Emb'][news_id])
                _news_category[nid] = _news_category_map[train_news_df['Category'][news_id]]
                _news_subcategory[nid] = _news_subcategory_map[train_news_df['SubCategory'][news_id]]
            elif news_id in dev_news_df.index:
                _embedding['news']['News_Title_Embedding'][nid] = torch.Tensor(dev_news_df['Title-Emb'][news_id])
                _embedding['news']['News_Abstract_Embedding'][nid] = torch.Tensor(dev_news_df['Abstract-Emb'][news_id])
                _news_category[nid] = _news_category_map[dev_news_df['Category'][news_id]]
                _news_subcategory[nid] = _news_subcategory_map[dev_news_df['SubCategory'][news_id]]
            if _embedding['news']['News_Title_Embedding'][nid].sum() == 0 or _embedding['news']['News_Abstract_Embedding'][nid].sum() == 0:
                _empty_news_emb += 1
        _embedding['news']['CateID'] = _news_category.long()  # cateid
        _embedding['news']['SubCateID'] = _news_subcategory.long()  # subcateid
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        category_embeddings = sentence_model.encode(_categories)
        subcategory_embeddings = sentence_model.encode(_subcategories)
        _embedding['news']['Category'] = torch.mm(torch.nn.functional.one_hot(_news_category.long()).float(), torch.Tensor(category_embeddings))
        _embedding['news']['SubCategory'] = torch.mm(torch.nn.functional.one_hot(_news_subcategory.long()).float(), torch.Tensor(subcategory_embeddings))
        print('Num of Empty News Embedding: {} in {}'.format(_empty_news_emb, len(_nodes['news'])))

        self._print('Processing Entity')
        # new entity id -> emb
        _embedding['entity']['Entity_Embedding'] = torch.zeros([self._num_node['entity'], 100])  # 100 as entity embedding dim
        for id, oid in enumerate(self._entity):  # WikidataID
            entity_wiki_id = 'Q{}'.format(oid)
            _embedding['entity']['Entity_Embedding'][id] = torch.Tensor(entity_embedding_dict[entity_wiki_id])

        self._print('Processing Word')
        _embedding['word']['Word_Embedding'] = torch.Tensor(word_emb)

        self._print('Processing User')
        _embedding['user']['Category'] = torch.ones([self._num_node['user'], len(_categories)]) * 1e-10  # avoid zero
        _embedding['user']['SubCategory'] = torch.ones([self._num_node['user'], len(_subcategories)]) * 1e-10
        for i in range(history_actions.shape[0]):
            uid = history_actions[i][0]
            nid = history_actions[i][1]
            news_id = self._news[nid]
            if news_id in train_news_df.index:
                cid = _news_category_map[train_news_df['Category'][news_id]]
                scid = _news_subcategory_map[train_news_df['SubCategory'][news_id]]
                _embedding['user']['Category'][uid, cid] += 1
                _embedding['user']['SubCategory'][uid, scid] += 1
            elif news_id in dev_news_df.index:
                cid = _news_category_map[dev_news_df['Category'][news_id]]
                scid = _news_subcategory_map[dev_news_df['SubCategory'][news_id]]
                _embedding['user']['Category'][uid, cid] += 1
                _embedding['user']['SubCategory'][uid, scid] += 1
        _embedding['user']['Category'] = torch.mm(_embedding['user']['Category'], torch.Tensor(category_embeddings))
        _embedding['user']['SubCategory'] = torch.mm(_embedding['user']['SubCategory'], torch.Tensor(subcategory_embeddings))

        # Mounting
        for ntype in _embedding:
            for emb_type in _embedding[ntype]:
                self._graph.nodes[ntype].data[emb_type] = _embedding[ntype][emb_type]
        
        # node2vec
        self._print("Perform training node2vec model")
        entity2vec, news2vec, user2vec, word2vec = node2vec(self, self.cfg['dataset_dir'], self._force_reload)
        self.graph.nodes['entity'].data['Node2Vec'] = entity2vec
        self.graph.nodes['news'].data['Node2Vec'] = news2vec
        self.graph.nodes['user'].data['Node2Vec'] = user2vec
        self.graph.nodes['word'].data['Node2Vec'] = word2vec

        # Edge Attr -- Time
        self._print('Time')
        self._graph.edges['pos_train'].data['Time'] = torch.Tensor(
            train_positive_actions.T[2]).type(torch.float)
        self._graph.edges['pos_dev'].data['Time'] = torch.Tensor(
            dev_positive_actions.T[2]).type(torch.float)
        self._graph.edges['neg_train'].data['Time'] = torch.Tensor(
            train_negative_actions.T[2]).type(torch.float)
        self._graph.edges['neg_dev'].data['Time'] = torch.Tensor(
            dev_negative_actions.T[2]).type(torch.float)
        self._graph.edges['pos_train_r'].data['Time'] = torch.Tensor(
            train_positive_actions.T[2]).type(torch.float)
        self._graph.edges['pos_dev_r'].data['Time'] = torch.Tensor(
            dev_positive_actions.T[2]).type(torch.float)
        self._graph.edges['neg_train_r'].data['Time'] = torch.Tensor(
            train_negative_actions.T[2]).type(torch.float)
        self._graph.edges['neg_dev_r'].data['Time'] = torch.Tensor(
            dev_negative_actions.T[2]).type(torch.float)

        # Edge Attr -- Label
        self._print('Label')
        self._graph.edges['pos_train'].data['Label'] = torch.ones(
            self._graph.edges['pos_train'].data['Time'].shape)
        self._graph.edges['pos_dev'].data['Label'] = torch.ones(
            self._graph.edges['pos_dev'].data['Time'].shape)
        self._graph.edges['neg_train'].data['Label'] = torch.zeros(
            self._graph.edges['neg_train'].data['Time'].shape)
        self._graph.edges['neg_dev'].data['Label'] = torch.zeros(
            self._graph.edges['neg_dev'].data['Time'].shape)
        self._graph.edges['pos_train_r'].data['Label'] = torch.ones(
            self._graph.edges['pos_train_r'].data['Time'].shape)
        self._graph.edges['pos_dev_r'].data['Label'] = torch.ones(
            self._graph.edges['pos_dev_r'].data['Time'].shape)
        self._graph.edges['neg_train_r'].data['Label'] = torch.zeros(
            self._graph.edges['neg_train_r'].data['Time'].shape)
        self._graph.edges['neg_dev_r'].data['Label'] = torch.zeros(
            self._graph.edges['neg_dev_r'].data['Time'].shape)
        
    def save_partition(self, sub_graph, id_map, edge_id_map, partition_dir):
        num_nodes = {ntype: sub_graph.num_nodes(ntype) for ntype in sub_graph.ntypes}
        num_links = {etype: sub_graph.num_edges(etype) for etype in sub_graph.etypes}
        
        reverse_edge_map = {k: {value: key for key, value in v.items()} for k, v in edge_id_map.items()}
        
        part_dev_session_positive = [
            [reverse_edge_map["pos_dev_r"][eid] for eid in pos_links if eid in reverse_edge_map["pos_dev_r"]]
            for pos_links in self._dev_session_positive
            if all(eid in reverse_edge_map["pos_dev_r"] for eid in pos_links)
        ]
        
        part_dev_session_negative = [
            [reverse_edge_map["neg_dev_r"][eid] for eid in neg_links  if eid in reverse_edge_map["neg_dev_r"]]
            for neg_links in self._dev_session_negative
            if all(eid in reverse_edge_map["neg_dev_r"] for eid in neg_links)
        ]
        
        part_train_sessions_positive = [
            [reverse_edge_map["pos_train"][eid] for eid in pos_links if eid in reverse_edge_map["pos_train"]]
            for pos_links in self._train_session_positive
            if all(eid in reverse_edge_map["pos_train"] for eid in pos_links)
        ]
        part_train_sessions_negative = [
            [reverse_edge_map["neg_train"][eid] for eid in neg_links if eid in reverse_edge_map["neg_train"]]
            for neg_links in self._train_session_negative
            if all(eid in reverse_edge_map["neg_train"] for eid in neg_links)
        ]
        
        graph_path = partition_dir + \
            '/graph_{}.json'.format(self.cfg['mind_version'])
        save_info_path = partition_dir + \
            '/info_{}.json'.format(self.cfg['mind_version'])
            
        if "s3://" in partition_dir:
            fs = s3fs.S3FileSystem(anon=False)
            
            temp_file = "temp.json"
            save_graphs(temp_file, [sub_graph])
            fs.put(temp_file, graph_path)
            
            info = {
                'num_node': json.dumps(num_nodes),
                'num_link': json.dumps(num_links),
                'reverse_etypes': json.dumps(self._reverse_etypes),
                'train_session_positive': seq_list(part_train_sessions_positive),
                'train_session_negative': seq_list(part_train_sessions_negative),
                'dev_session_positive': seq_list(part_dev_session_positive),
                'dev_session_negative': seq_list(part_dev_session_negative),
                'id_map': json.dumps(id_map),
            }
            pickle.dump(info, fs.open(save_info_path, 'wb'))
            
            os.remove(temp_file)
            
        else:
            save_graphs(graph_path, [sub_graph])
            save_info(save_info_path, {
                'num_node': json.dumps(num_nodes),
                'num_link': json.dumps(num_links),
                'reverse_etypes': json.dumps(self._reverse_etypes),
                'train_session_positive': seq_list(part_train_sessions_positive),
                'train_session_negative': seq_list(part_train_sessions_negative),
                'dev_session_positive': seq_list(part_dev_session_positive),
                'dev_session_negative': seq_list(part_dev_session_negative),
                'id_map': json.dumps(id_map),
            })   
        

    def save(self):
        r"""
        保存图和标签
        """
        save_graphs(self._graph_path, [self._graph])
        save_info(self._save_info, {
            'num_node': json.dumps(self._num_node),
            'num_link': json.dumps(self._num_link),
            'reverse_etypes': json.dumps(self._reverse_etypes),
            'train_session_positive': seq_list(self._train_session_positive),
            'train_session_negative': seq_list(self._train_session_negative),
            'dev_session_positive': seq_list(self._dev_session_positive),
            'dev_session_negative': seq_list(self._dev_session_negative),
        })

    def load(self):
        r"""
         从目录 `self.save_path` 里读取处理过的数据
        """
        graphs, label_dict = load_graphs(self._graph_path)
        self._graph = graphs[0]
        info = load_info(self._save_info)
        self._num_node = json.loads(info['num_node'])
        self._num_link = json.loads(info['num_link'])
        self._reverse_etypes = json.loads(info['reverse_etypes'])
        self._train_session_positive = unseq_list(info['train_session_positive'])
        self._train_session_negative = unseq_list(info['train_session_negative'])
        self._dev_session_positive = unseq_list(info['dev_session_positive'])
        self._dev_session_negative = unseq_list(info['dev_session_negative'])
        if 'id_map' in info:
            self._id_map = json.loads(info['id_map'])
   

    def has_cache(self):
        # 检查在 `self.save_path` 里是否有处理过的数据文件
        return os.path.exists(self._graph_path) and os.path.exists(self._save_info)

    def _print(self, msg):
        if self._verbose:
            print('[{}] @ [{}] \n>>> {}'.format(self._name,
                                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg))

    def print_statistics(self):
        self._print('num_node: {}\nnum_relation: {}'.format(
            json.dumps(self._num_node), json.dumps(self._num_link)))

    @property
    def num_node(self):
        return self._num_node

    @property
    def num_relation(self):
        return self._num_link

    @property
    def reverse_etypes(self):
        return self._reverse_etypes

    @property
    def graph(self):
        return self._graph

    def get_gnn_train_loader(self, base_etypes, num_layers):
        train_sampled_datasets = building_training_dataset(self._train_session_positive, self._train_session_negative, self.cfg["gnn_neg_ratio"])
        pos_edges = torch.Tensor(train_sampled_datasets[:, 0]).type(torch.int32)
        neg_edges = torch.Tensor(train_sampled_datasets[:, 1:]).type(torch.int32)  

        shuffled_id = np.array(list(range(pos_edges.shape[0])))
        np.random.shuffle(shuffled_id)
        pos_edges_shuffled = pos_edges[shuffled_id]
        neg_edges_shuffled = neg_edges[shuffled_id]

        # train_sampler = "MultiLayerFull"
        if self.cfg['train_sampler'] == 'MultiLayer':
            sampler = dgl.dataloading.MultiLayerNeighborSampler(self.cfg['train_sampler_param'], replace=False)
        elif self.cfg['train_sampler'] == 'SimilarityWeightedMultiLayer' or self.cfg['train_sampler'] == 'InverseDegreeWeightedMultiLayer':
            sampler = dgl.dataloading.MultiLayerNeighborSampler(self.cfg['train_sampler_param'], prob='Sampling_Weight', replace=False)
        elif self.cfg['train_sampler'] == 'MultiLayerFull':
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
        else:
            raise Exception('Unexpected Neighbor Sampler')

        base_graph = self._graph.edge_type_subgraph(base_etypes)

        pos_collator = dgl.dataloading.EdgeCollator(
            g=self._graph, eids={'pos_train': pos_edges_shuffled}, graph_sampler=sampler, g_sampling=base_graph
        )
        pos_dataloader = DataLoader(pos_collator.dataset, collate_fn=pos_collator.collate,
                                    batch_size=len(pos_collator.dataset), shuffle=False,
                                    drop_last=False,  num_workers=0)
                                    # batch_size=hparams['batch_size'], shuffle=False,

        neg_collator = dgl.dataloading.EdgeCollator(
            g=self._graph, eids={'neg_train': neg_edges_shuffled.flatten(start_dim=0)}, graph_sampler=sampler, g_sampling=base_graph,
        )
        neg_dataloader = DataLoader(neg_collator.dataset, collate_fn=neg_collator.collate,
                                    batch_size=len(neg_collator.dataset) * self.cfg['gnn_neg_ratio'], shuffle=False,
                                    drop_last=False,  num_workers=0)
                                    # batch_size=hparams['batch_size'] * hparams['gnn_neg_ratio'], 
                                    
        return pos_dataloader, neg_dataloader

    def get_dev_session_loader(self, shuffle):
        dev_dataset = SessionDataset(self._dev_session_positive, self._dev_session_negative)
        # shuffle=False if Decaying is activted
        return DataLoader(dev_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    
    def get_val_session_loader(self, size):
        if self._indices == None:
            self._indices = random.sample(range(len(self._dev_session_positive)), size)
        dev_dataset = SessionDataset([self._dev_session_positive[i] for i in self._indices], [self._dev_session_negative[i] for i in self._indices])
        
        # shuffle=False if Decaying is activted
        return DataLoader(dev_dataset, batch_size=1, shuffle=True, num_workers=0)

    def get_gnn_dev_node_loader(self, etypes, num_layers):  # Dev - for Generate Node Representations
        pos_edges = torch.Tensor(list(range(self._num_link['pos_dev_r']))).type(torch.int32)
        neg_edges = torch.Tensor(list(range(self._num_link['neg_dev_r']))).type(torch.int32)
        sub_g = dgl.edge_subgraph(self._graph, {('news', 'pos_dev_r', 'user'): pos_edges, ('news', 'neg_dev_r', 'user'): neg_edges})

        base_graph = self._graph.edge_type_subgraph(etypes)
        


        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)


        user_collator = dgl.dataloading.NodeCollator(base_graph, {'user': sub_g.dstdata['_ID']['user']}, sampler)
        news_collator = dgl.dataloading.NodeCollator(base_graph, {'news': sub_g.srcdata['_ID']['news']}, sampler)
        
        user_dataloader = torch.utils.data.DataLoader(
            user_collator.dataset, collate_fn=user_collator.collate,
            batch_size=len(user_collator.dataset), shuffle=True, drop_last=False, num_workers=0
        )
        news_dataloader = torch.utils.data.DataLoader(
            news_collator.dataset, collate_fn=news_collator.collate,
            batch_size=len(news_collator.dataset), shuffle=True, drop_last=False, num_workers=0
        )
        return user_dataloader, news_dataloader


class SessionDataset(Dataset):
    def __init__(self, pos_links, neg_links):
        self.pos_links = pos_links
        self.neg_links = neg_links

    def __getitem__(self, index):
        return self.pos_links[index], self.neg_links[index]

    def __len__(self):
        return len(self.pos_links)


class BaseDataset(Dataset):
    def __init__(self, v):
        self.v = v

    def __getitem__(self, index):
        return self.v[index]

    def __len__(self):
        return len(self.v)
