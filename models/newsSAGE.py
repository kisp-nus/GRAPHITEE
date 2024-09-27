"""Implement simple gnn model"""

from typing import Optional
import torch
from torch import nn
from models.base_model import BaseGNN
from dgl.nn.pytorch import HeteroGraphConv, SAGEConv
import torch.nn.functional as F


from models.utils.layers import MultiLayerProcessorAdaptor, ScorePredictor, attention, kl


class NewsSAGEModel(BaseGNN):
    """Implement a GNN model

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output feature dimension
    conv_layer : Optional[nn.Module], optional
        Convolution layer, by default GCNConv
    n_layers : int, optional
        Number of layers, by default 2
    activation : nn.Module, optional
        Activation function, by default nn.ReLU()
    dropout : float, optional
        Dropout rate, by default 0.0
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feat_hidden: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        emb_data: None,
        device,
        n_layers: int,
        cross_score: bool,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.cross_score = cross_score
        self.device = device
        self.node_emb_data = emb_data
        self.hetero_convs = nn.ModuleList()
        for _ in range(n_layers):
            self.hetero_convs.append(HeteroGraphConv({
            'history': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'history_r': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'ne_link': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'ne_link_r': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'ue_link': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'ue_link_r': SAGEConv(hidden_dim, hidden_dim, 'mean'),
        }))


        self.dropout_rate = dropout        
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        self.attr_set = {}
        self.adaptor_align = nn.ModuleDict()
        for node_type in emb_data:
            for emb_type in emb_data[node_type]:
                if emb_type in self.adaptor_align:
                    continue
                else:
                    self.adaptor_align[emb_type] = MultiLayerProcessorAdaptor([emb_data[node_type][emb_type], feat_hidden, input_dim])
                    self.attr_set[emb_type] = len(self.attr_set)
                    
        self.fusioner = nn.Sequential(
            nn.Linear(len(self.attr_set) * input_dim, 2 * input_dim),
            nn.Linear(2 * input_dim, input_dim),
        )
        fusioner_router = {}
        for node_type in self.node_emb_data:
            fusioner_router[node_type] = torch.zeros([len(self.attr_set), len(self.node_emb_data[node_type])])  # ALL_ATTR_NUM * CUR_NODE_ATTR_NUM
            for i, emb_type in enumerate(self.node_emb_data[node_type]):
                fusioner_router[node_type][self.attr_set[emb_type]][i] = 1
        self.fusioner_router = nn.ParameterDict({
            node_type: nn.Parameter(fusioner_router[node_type])
            for node_type in fusioner_router
        }).to(device)
        
        # Attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=384, num_heads=1, batch_first=True)
        self.projection_layer = nn.Linear(384, hidden_dim)
        
        self.projection_layer = nn.ModuleDict({
            node_type: nn.Linear(384, hidden_dim)
            for node_type in emb_data
        })
        
        # self.batch_norms = nn.ModuleDict({
        #     node_type: nn.BatchNorm1d(output_dim * 2) 
        #     for node_type in emb_data
        # })
        
        self.denser = nn.Linear(self.input_dim, self.output_dim * 2)
        self.scorer = ScorePredictor(output_dim, device, cross_score=cross_score)
        
    def adapt(self, blocks):
        input_features = {}
        for node_type in self.node_emb_data:
            node_attr = []
            for emb_type in self.node_emb_data[node_type]:
                # Directly fetch features from blocks[0].srcdata without using embeddings
                node_attr.append(self.adaptor_align[emb_type](
                    blocks[0].srcdata[emb_type][node_type].to(self.device)
                ).unsqueeze(1))
            node_attr = torch.cat(node_attr, dim=1)
            node_attr, _ = attention(node_attr, node_attr, node_attr, self.device)
            input_features[node_type] = node_attr
        return input_features
        
    def fusion(self, adapted_features):
        input_features = {}
        for node_type in adapted_features:
            if adapted_features[node_type].shape[0] == 0:
                continue
            else:
                input_features[node_type] = self.fusioner(
                    torch.matmul(self.fusioner_router[node_type], adapted_features[node_type]).reshape(adapted_features[node_type].shape[0], -1)
                )
        return input_features
    
    def forward(self, edge_subgraph, blocks, scoring_edge):
        adapted_features = self.adapt(blocks)
        input_features = self.fusion(adapted_features)
        
        for i in range(len(self.hetero_convs)):
            conv = self.hetero_convs[i]
            input_features = conv(blocks[i], input_features)
            # average GAT heads:
            # input_features["user"] = input_features["user"].mean(dim=1)
            input_features = {k: F.relu(v) for k, v in input_features.items()}
            if i != len(self.hetero_convs) - 1:
                input_features = {k: self.dropout(v) for k, v in input_features.items()}
            
            
        # output_features, _ = self.rgcn(blocks, input_features)
        output_features = input_features
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :self.output_dim], 
                output_features[node_type][:, self.output_dim:]
            ))
        return self.scorer(edge_subgraph, output_features, scoring_edge), output_features, kls
    
    def encode(self, blocks, for_prediction=False):
        adapted_features = self.adapt(blocks)
        input_features = self.fusion(adapted_features)
        
        for i in range(len(self.hetero_convs)):
            conv = self.hetero_convs[i]
            input_features = conv(blocks[i], input_features)
            # Average GAT heads
            # input_features["user"] = input_features["user"].mean(dim=1)
            input_features = {k: F.relu(v) for k, v in input_features.items()}
            
        output_features = input_features
            
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :self.output_dim], 
                output_features[node_type][:, self.output_dim:]
            ))
        return output_features
    
    def get_nth_layer(self, n: int) -> nn.Module:
        """Get the nth layer of the model

        Parameters
        ----------
        n : int
            Index of the layer

        Returns
        -------
        nn.Module
            The nth layer, object with forward method
        """
        batch_norms = nn.ModuleDict({
            node_type: nn.BatchNorm1d(self.output_dim * 2) 
            for node_type in self.node_emb_data
        })
        denser = nn.Linear(self.input_dim, self.output_dim * 2)
        scorer = ScorePredictor(self.output_dim, self.device, cross_score=self.cross_score)
        
        # first layer (no GNN)
        if n == 0:
            return IntermediateModel(
                self.device,
                self.node_emb_data,
                # self.projection_layer,
                # self.attention_layer,
                self.adaptor_align, 
                self.fusioner_router, 
                self.fusioner, 
                None, 
                self.dropout,
                denser, 
                scorer
                )
        # last GNN layer
        if n == len(self.hetero_convs):
            return IntermediateModel(
                self.device,
                self.node_emb_data,
                # self.projection_layer,
                # self.attention_layer,
                None, 
                None,
                None,
                self.hetero_convs[n - 1],
                self.dropout,
                self.denser,
                self.scorer,
                isLast=True
                )
            
        # intermediate GNN layer
        return IntermediateModel(
            self.device,
            self.node_emb_data,
            # self.projection_layer,
            # self.attention_layer,
            None,
            None,
            None,
            self.hetero_convs[n-1],
            self.dropout,
            denser,
            scorer            
            )
        

class IntermediateModel(nn.Module):
    """Model with one aggregation layer and multiple following layers"""

    def __init__(self, device, node_emb_data, 
                # projection_layer,
                # attention_layer,
                 adaptor_align,
                 fusioner_router, 
                 fusioner,
                 conv, dropout, denser, scorer, isLast=False) -> None:
        super().__init__()
        self.device = device
        self.node_emb_data = node_emb_data
        # self.projection_layer = projection_layer
        # self.attention_layer = attention_layer
        self.adaptor_align = adaptor_align
        self.fusioner_router = fusioner_router
        self.fusioner = fusioner
        self.hetero_conv = conv
        self.dropout = dropout
        self.denser = denser
        self.scorer = scorer
        self.isLast = isLast

    
    def adapt(self, blocks, encode_source=True):
        input_features = {}
        for node_type in self.node_emb_data:
            node_attr = []
            for emb_type in self.node_emb_data[node_type]:
                # Directly fetch features from blocks[0].srcdata without using embeddings
                if encode_source:
                    node_attr.append(self.adaptor_align[emb_type](
                        blocks[0].srcdata[emb_type][node_type].to(self.device)
                    ).unsqueeze(1))
                else:
                    node_attr.append(self.adaptor_align[emb_type](
                        blocks[0].dstdata[emb_type][node_type].to(self.device) #todo
                    ).unsqueeze(1))
            node_attr = torch.cat(node_attr, dim=1)
            node_attr, _ = attention(node_attr, node_attr, node_attr, self.device)
            input_features[node_type] = node_attr
        return input_features
    
    def fusion(self, adapted_features):
        input_features = {}
        for node_type in adapted_features:
            if adapted_features[node_type].shape[0] == 0:
                continue
            else:
                input_features[node_type] = self.fusioner(
                    torch.matmul(self.fusioner_router[node_type], adapted_features[node_type]).reshape(adapted_features[node_type].shape[0], -1)
                )
        return input_features

    def adapt_attention(self, blocks, encode_source=True):
        input_features = {}
        for node_type in self.node_emb_data:
            node_attr = []
            for emb_type in self.node_emb_data[node_type]:
                if encode_source:
                    feature_data = blocks[0].srcdata[emb_type][node_type].to(self.device)
                else:
                    feature_data = blocks[0].dstdata[emb_type][node_type].to(self.device)
                node_attr.append(feature_data.unsqueeze(1))                
            node_attr = torch.cat(node_attr, dim=1)
            node_attr, _ = self.attention_layer(node_attr, node_attr, node_attr)
            node_attr = self.projection_layer[node_type](node_attr)
            input_features[node_type] = node_attr.mean(1)
        return input_features

    def forward(self, edge_subgraph, blocks, scoring_edge, input_features=None):
        """Forward pass"""
        if self.hetero_conv is None:
            # adapted_features = self.adapt_attention(blocks, encode_source=False)
            # input_features = adapted_features
            adapted_features = self.adapt(blocks, encode_source=False)
            # print("adapted")
            input_features = self.fusion(adapted_features)
            
            # print("fused")
        else:
            assert input_features is not None
            
            input_features = self.hetero_conv(blocks[0], input_features)
            input_features = {k: F.relu(v) for k, v in input_features.items()}
            # if self.dropout is not None:
            input_features = {k: self.dropout(v) for k, v in input_features.items()}
            
            
        output_features = input_features
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
            # output_features[node_type] = self.batch_norms[node_type](output_features[node_type]) 
            output_features[node_type] = self.dropout(output_features[node_type])
            
        # print("densed and dropped")
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :self.denser.in_features], 
                output_features[node_type][:, self.denser.in_features:]
            ))
        # print("scoring")
        return self.scorer(edge_subgraph, output_features, scoring_edge), {k: v.detach() for k, v in output_features.items()}, kls

    def encode(self, blocks, input_features=None, for_prediction=False, encode_source=True):
        if self.hetero_conv is None:
            # adapted_features = self.adapt_attention(blocks, encode_source=True)
            # input_features = adapted_features
            adapted_features = self.adapt(blocks, encode_source=not for_prediction)
            input_features = self.fusion(adapted_features)
            
        
            # TODO for first layer should we still apply the denser?
            if not for_prediction:
                return {k: v.detach() for k, v in input_features.items()}

        else:
            assert input_features is not None
            input_features =  self.hetero_conv(blocks[0], input_features)
            input_features = {k: F.relu(v) for k, v in input_features.items()}
        
        if not self.isLast and not for_prediction: 
            # intermediate layer, skip denser
            return {k: v.detach() for k, v in input_features.items()}
        
        # if self.dropout is not None:
        #     input_features = {k: self.dropout(v) for k, v in input_features.items()}
            
            
        output_features = input_features
            
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
            # output_features[node_type] = self.batch_norms[node_type](output_features[node_type])  # Apply batch normalization
            output_features[node_type] = self.dropout(output_features[node_type])
            
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :self.denser.in_features], 
                output_features[node_type][:, self.denser.in_features:]
            ))
        return output_features
