import argparse
from pickle import FALSE

import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv

class ILM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(ILM, self).__init__()

        self.iterations = 3
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if num_layers == 1:
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(in_channels, out_channels, heads=head))

        elif num_layers > 1:
            hidden_channels= int(self.hidden_channels/head)
            self.convs.append(GATConv(in_channels, hidden_channels, heads=head))
            
            for _ in range(num_layers - 2):
                hidden_channels =  int(self.hidden_channels/head)
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels, heads=head))
            
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(hidden_channels, out_channels, heads=head))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in gat: ', len(self.convs))
            self.invest = 0

        # Prop layers
        props = []
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            #emb = torch.tanh(F.leaky_relu(x))
            props.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last prop layer
        x = self.convs[-1](x, adj_t)
        props.append(x)
        #props.append(torch.tanh(F.leaky_relu(x)))

        local_emb = props[0]
        global_emb = props[-1]

        return local_emb, global_emb
