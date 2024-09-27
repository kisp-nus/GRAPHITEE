import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from torch.autograd import Variable
import torch.nn.functional as F

from copy import deepcopy
import math

class MultiLayerProcessorAdaptor(nn.Module):
    def __init__(self, features):
        super().__init__()
        if len(features) < 2:
            raise Exception('At least 2 is needed to build MultiLayerProcessorAdaptor')
        self.adaptor = nn.ModuleList()
        for i in range(len(features)-1):
            self.adaptor.append(nn.Linear(features[i], features[i+1]))
    
    def forward(self, x):
        x = x.type(torch.float)
        for i in range(len(self.adaptor)):
            x = self.adaptor[i](x)
        return x

class ScorePredictor(nn.Module):
    def __init__(self, out_features, device, cross_score=False, sample=0):
        super().__init__()
        self.out_features = out_features
        self.sample = sample
        self.cross_score = cross_score
        self.device = device
        if cross_score:
            self.cross = nn.Sequential(
                nn.Linear(2 * self.out_features, 2 * self.out_features),
                nn.Linear(2 * self.out_features, 2 * self.out_features),
            )
    
    def get_representation(self, nodes):
        if self.sample <= 0:
            representation = nodes.data['GNN_Emb'][:, :self.out_features]
            return {'Representation': representation}
        else:
            representation = []
            for i in range(self.sample):
                sub_representation = reparametrize(nodes.data['GNN_Emb'][:, :self.out_features], nodes.data['GNN_Emb'][:, self.out_features:])
                representation.append(sub_representation)
            return {'Representation': torch.cat(representation, dim=-1)}

    def msgfunc_score_neg(self, edges):
        src_emb = edges.src['Representation']
        dst_emb = edges.dst['Representation']
        if self.cross_score: 
            src_emb = src_emb.to(self.device)
            dst_emb = dst_emb.to(self.device)
            if self.sample <= 0:
                crossed_emb = self.cross(torch.cat([src_emb, dst_emb], 1))
                crossed_src_emb = crossed_emb[:, :self.out_features]
                crossed_dst_emb = crossed_emb[:, self.out_features:]
                score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
            else:
                scores = []
                for i in range(self.sample):
                    sub_src_emb = src_emb[i*self.out_features : (i+1)*self.out_features]
                    sub_dst_emb = dst_emb[i*self.out_features : (i+1)*self.out_features]
                    crossed_emb = self.cross(torch.cat([sub_src_emb, sub_dst_emb], 1))
                    crossed_src_emb = crossed_emb[:, :self.out_features]
                    crossed_dst_emb = crossed_emb[:, self.out_features:]
                    score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach()#.cpu()
                    scores.append(score)
                score = sum(scores) / len(scores)
        else:
            if self.sample<= 0:
                score = (src_emb * dst_emb).sum(dim=1)
            else:
                score = (src_emb * dst_emb).sum(dim=1) / self.sample
        return {'neg_score': score, 'neg_news_representation': src_emb.cpu()}

    def msgfunc_score_pos(self, edges):
        src_emb = edges.src['Representation']
        dst_emb = edges.dst['Representation']
        if self.cross_score: 
            src_emb = src_emb.to(self.device)
            dst_emb = dst_emb.to(self.device)
            if self.sample <= 0:
                crossed_emb = self.cross(torch.cat([src_emb, dst_emb], 1))
                crossed_src_emb = crossed_emb[:, :self.out_features]
                crossed_dst_emb = crossed_emb[:, self.out_features:]
                score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
            else:
                scores = []
                for i in range(self.sample):
                    sub_src_emb = src_emb[i*self.out_features : (i+1)*self.out_features]
                    sub_dst_emb = dst_emb[i*self.out_features : (i+1)*self.out_features]
                    crossed_emb = self.cross(torch.cat([sub_src_emb, sub_dst_emb], 1))
                    crossed_src_emb = crossed_emb[:, :self.out_features]
                    crossed_dst_emb = crossed_emb[:, self.out_features:]
                    score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach()#.cpu()
                    scores.append(score)
                score = sum(scores) / len(scores)
        else:
            if self.sample <= 0:
                score = (src_emb * dst_emb).sum(dim=1)
            else:
                score = (src_emb * dst_emb).sum(dim=1) / self.sample
        return {'pos_score': score, 'pos_news_representation': src_emb.cpu()}

    def msgfunc_score_neg_edc(self, edges):
        src_emb = edges.src['Representation']
        # dst_emb = edges.dst['Representation']
        Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
        News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
        for i, cate in enumerate(edges.src['CateID']):
            Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
            News_Pref[i] = edges.dst['News_Pref'][i][cate]
        # delta-t and decay
        delta_t = edges.data['Time'][0] - Last_Update_Time
        decaying_weight = self.get_decay_weight(delta_t)
        # reparametrize pref
        pref_shape = News_Pref.shape[:-1]
        decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
        distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
        if self.sample <= 0:
            pref_emb = distribution_decayed_pref.reshape(-1, 2 * self.out_features)[:, :self.out_features].reshape(pref_shape[0], pref_shape[1], self.out_features)
        else:
            pref_emb = []
            for i in range(self.sample):
                sub_pref_emb = reparametrize(
                        mu=distribution_decayed_pref.reshape(-1, 2 * self.out_features)[:, :self.out_features],
                        logvar=distribution_decayed_pref.reshape(-1, 2 * self.out_features)[:, self.out_features:]
                    ).reshape(pref_shape[0], pref_shape[1], self.out_features)
                pref_emb.append(sub_pref_emb)
            pref_emb = torch.cat(pref_emb, dim=-1)
        dst_pref_emb, _ = attention(src_emb.unsqueeze(1), pref_emb, pref_emb, self.device)
        dst_pref_emb = dst_pref_emb.squeeze(1)
        return {
            'neg_score': (src_emb * dst_pref_emb).sum(dim=1), 
            'neg_news_representation': src_emb,
            'src_repr': edges.src['GNN_Emb'], 
            'time': edges.data['Time'],
            'cate': edges.src['CateID']
        }

    def msgfunc_score_pos_edc(self, edges):
        src_emb = edges.src['Representation']
        # dst_emb = edges.dst['Representation']
        Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
        News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
        for i, cate in enumerate(edges.src['CateID']):
            Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
            News_Pref[i] = edges.dst['News_Pref'][i][cate]
        # delta-t and decay
        delta_t = edges.data['Time'][0] - Last_Update_Time
        decaying_weight = self.get_decay_weight(delta_t)
        # reparametrize pref
        pref_shape = News_Pref.shape[:-1]
        decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
        distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
        if self.sample <= 0:
            pref_emb = distribution_decayed_pref.reshape(-1, 2 * self.out_features)[:, :self.out_features].reshape(pref_shape[0], pref_shape[1], self.out_features)
        else:
            pref_emb = []
            for i in range(self.sample):
                sub_pref_emb = reparametrize(
                        mu=distribution_decayed_pref.reshape(-1, 2 * self.out_features)[:, :self.out_features],
                        logvar=distribution_decayed_pref.reshape(-1, 2 * self.out_features)[:, self.out_features:]
                    ).reshape(pref_shape[0], pref_shape[1], self.out_features)
                pref_emb.append(sub_pref_emb)
            pref_emb = torch.cat(pref_emb, dim=-1)
        dst_pref_emb, _ = attention(src_emb.unsqueeze(1), pref_emb, pref_emb, self.device)
        dst_pref_emb = dst_pref_emb.squeeze(1)
        return {
            'pos_score': (src_emb * dst_pref_emb).sum(dim=1), 
            'pos_news_representation': src_emb,
            'src_repr': edges.src['GNN_Emb'], 
            'time': edges.data['Time'],
            'cate': edges.src['CateID']
        }
    
    def msgfunc_score_vgnn(self, edges):  # for training
        src_emb = edges.src['Representation']
        dst_emb = edges.dst['Representation']
        if self.cross_score: 
            src_emb = src_emb.to(self.device)
            dst_emb = dst_emb.to(self.device)
            if self.sample <= 0:
                crossed_emb = self.cross(torch.cat([src_emb, dst_emb], 1))
                crossed_src_emb = crossed_emb[:, :self.out_features]
                crossed_dst_emb = crossed_emb[:, self.out_features:]
                score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach()#.cpu()
            else:
                scores = []
                for i in range(self.sample):
                    sub_src_emb = src_emb[i*self.out_features : (i+1)*self.out_features]
                    sub_dst_emb = dst_emb[i*self.out_features : (i+1)*self.out_features]
                    crossed_emb = self.cross(torch.cat([sub_src_emb, sub_dst_emb], 1))
                    crossed_src_emb = crossed_emb[:, :self.out_features]
                    crossed_dst_emb = crossed_emb[:, self.out_features:]
                    score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach()#.cpu()
                    scores.append(score)
                score = sum(scores) / len(scores)
        else:
            if self.sample <= 0:
                score = (src_emb * dst_emb).sum(dim=1)
            else:
                score = (src_emb * dst_emb).sum(dim=1) / self.sample
        return {'score': score}

    def reduce_score_pos(self, nodes):
        pos_score = nodes.mailbox['pos_score']
        pos_news_representation = nodes.mailbox['pos_news_representation']
        return {'pos_score': pos_score, 'pos_news_representation': pos_news_representation}

    def reduce_score_neg(self, nodes):
        neg_score = nodes.mailbox['neg_score']
        neg_news_representation = nodes.mailbox['neg_news_representation']
        return {'neg_score': neg_score, 'neg_news_representation': neg_news_representation}

    def reduce_score_pos_edc(self, nodes):
        pos_score = nodes.mailbox['pos_score']
        pos_news_representation = nodes.mailbox['pos_news_representation']
        src_repr = nodes.mailbox['src_repr']
        time = nodes.mailbox['time']
        cate = nodes.mailbox['cate']
        new_pref = deepcopy(nodes.data['News_Pref'])
        new_lut = deepcopy(nodes.data['Last_Update_Time'])
        for dst_node in range(src_repr.shape[0]):
            for src_node in range(src_repr.shape[1]):
                i = new_lut[dst_node][cate[dst_node][src_node]].argmin()
                new_pref[dst_node][cate[dst_node][src_node]][i] = src_repr[dst_node][src_node]
                new_lut[dst_node][cate[dst_node][src_node]][i] = time[dst_node][src_node]
        return {
            'pos_score': pos_score, 
            'pos_news_representation': pos_news_representation, 
            'pref': new_pref, 
            'lut': new_lut
        }

    def reduce_score_neg_edc(self, nodes):
        neg_score = nodes.mailbox['neg_score']
        neg_news_representation = nodes.mailbox['neg_news_representation']
        src_repr = nodes.mailbox['src_repr']
        time = nodes.mailbox['time']
        cate = nodes.mailbox['cate']
        new_pref = deepcopy(nodes.data['News_Pref'])
        new_lut = deepcopy(nodes.data['Last_Update_Time'])
        for dst_node in range(src_repr.shape[0]):
            for src_node in range(src_repr.shape[1]):
                i = new_lut[dst_node][cate[dst_node][src_node]].argmin()
                new_pref[dst_node][cate[dst_node][src_node]][i] = src_repr[dst_node][src_node]
                new_lut[dst_node][cate[dst_node][src_node]][i] = time[dst_node][src_node]
        return {
            'neg_score': neg_score, 
            'neg_news_representation': neg_news_representation, 
            'pref': new_pref, 
            'lut': new_lut
        }
    
    def get_decay_weight(self, delta_t):
        shape = delta_t.shape
        return torch.Tensor([math.exp(- 0.2* math.pow(dt, 0.25)) for dt in delta_t.reshape(-1)]).reshape(shape)

    def forward(self, edge_subgraph, x, scoring_edge):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['GNN_Emb'] = x
            edge_subgraph.apply_nodes(self.get_representation, ntype='user')
            edge_subgraph.apply_nodes(self.get_representation, ntype='news')
            edge_subgraph.apply_edges(self.msgfunc_score_vgnn, etype=scoring_edge)
            return edge_subgraph.edata['score'][scoring_edge]



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def kl(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


def kl_gnn(mu):
    return torch.mean(-0.5 * torch.sum(- mu ** 2, dim=1), dim=0)


def attention(query, key, value, device, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(Variable(mask) == 0, -1e9).to(device)
    p_attn = F.softmax(scores, dim=-1)  # torch.Size([1326, 8, 20, 20])
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    import dgl
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    dgl.seed(seed)