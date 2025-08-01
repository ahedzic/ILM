import argparse
from pickle import FALSE

import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import RGATConv, RGCNConv

# from logger import Logger
from torch.nn import Embedding
# from utils import init_seed, get_param
from torch.nn.init import xavier_normal_
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)
from torch_geometric.nn import global_sort_pool
import math

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(RGCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))

        elif num_layers > 1:
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
                
            for _ in range(num_layers - 2):
                self.convs.append(
                    RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t, edge_types):
        props = []
        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_types)
            props.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_types)
        props.append(x)
        return props


class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, out_channels, num_layers,
                 dropout, mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(RGAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if num_layers == 1:
            out_channels = int(self.out_channels/head)
            self.convs.append(RGATConv(in_channels, out_channels, num_relations, heads=head))

        elif num_layers > 1:
            hidden_channels= int(self.hidden_channels/head)
            self.convs.append(RGATConv(in_channels, hidden_channels, num_relations, heads=head))
            
            for _ in range(num_layers - 2):
                hidden_channels =  int(self.hidden_channels/head)
                self.convs.append(
                    RGATConv(hidden_channels, hidden_channels, num_relations, heads=head))
            
            out_channels = int(self.out_channels/head)
            self.convs.append(RGATConv(hidden_channels, out_channels, num_relations, heads=head))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t, edge_types):
        props = []

        if self.invest == 1:
            print('layers in gat: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_types)
            props.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_types)
        props.append(x)
        return props
