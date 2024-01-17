import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EdgeScore(nn.Module):
    def __init__(self, h_feats, num_edge_types):
        super().__init__()
        self.W1 = nn.Linear(h_feats, int(h_feats / 2))
        self.W2 = nn.Linear(int(h_feats / 2), num_edge_types)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.num_edge_types = num_edge_types
        self.add_edge_1 = nn.Linear(h_feats, int(h_feats / 2))
        self.add_edge_2 = nn.Linear(int(h_feats / 2), 1)
        self.batchnorm = nn.BatchNorm1d(int(h_feats / 2))
        self.batchnorm2 = nn.BatchNorm1d(num_edge_types)
        self.batchnorm_add = nn.BatchNorm1d(int(h_feats / 2))
        self.batchnorm_add2 = nn.BatchNorm1d(1)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W1.weight, gain=gain)
        nn.init.xavier_normal_(self.W2.weight, gain=gain)
        nn.init.xavier_normal_(self.add_edge_1.weight, gain=gain)
        nn.init.xavier_normal_(self.add_edge_2.weight, gain=gain)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        
        if self.num_edge_types > 1:
            scores = self.softmax(self.batchnorm2(self.W2(F.relu(self.batchnorm(self.W1(h.float()))))))
        else:
            scores = self.sigmoid(self.batchnorm2(self.W2(F.relu(self.batchnorm(self.W1(h.float()))))))
            empty = torch.zeros(scores.shape[0]).to('cuda:' + str(scores.get_device()))

        if self.num_edge_types > 1:
            return {'score': scores, 'add_edge_score': self.sigmoid(self.batchnorm_add2(self.add_edge_2(F.leaky_relu(self.batchnorm_add(self.add_edge_1(h))))))}
        else:
            return {'score': scores, 'add_edge_score': empty}

    def forward(self, g):
        with g.local_scope():
            g.ndata['h'] = g.ndata['feat']
            g.apply_edges(self.apply_edges)
            
            return g.edata['score'], g.edata['add_edge_score']
