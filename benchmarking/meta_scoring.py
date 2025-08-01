
import torch
import torch.nn.functional as F

import torch.nn as nn

class meta_score(torch.nn.Module):
    def __init__(self, feat_channels, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, weights, device):
        super(meta_score, self).__init__()

        self.device = device
        self.weights = weights
        self.late_meta_attn_fc_1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.late_meta_attn_fc_2 = nn.Linear(hidden_channels, 3)
        input_size = in_channels
        self.feat_layer = nn.Linear(feat_channels, in_channels)
        self.add_edge = nn.Linear(1, 1)

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_meta_distributions()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.late_meta_attn_fc_1.weight, gain=gain)

        for lin in self.lins:
            lin.reset_parameters()

    def reset_meta_distributions(self):
        self.meta_distributions = torch.zeros(3).to(self.device)
        self.meta_count = 0

    def get_meta_distributions(self):
        return self.meta_distributions / self.meta_count

    def forward(self, x, h, edge_i, edge_j, iteration=0, density=None, clustering_coeffs=None, graph_clustering_coeff=None):
        raw_feat = x[edge_i] * x[edge_j]
        feat = self.feat_layer(x[edge_i] * x[edge_j])
        feat = torch.nn.functional.normalize(feat, dim=1)
        glob = torch.sum(h[-1], dim=0)
        glob = torch.nn.functional.normalize(glob.view(-1, glob.shape[0]))
        glob_all = glob.repeat(len(h[0]), 1)
        glob = glob.repeat(len(edge_i), 1)
        local = h[-1][edge_i] * h[-1][edge_j]
        local = torch.nn.functional.normalize(local, dim=1)
        diff = abs(h[0][edge_i] - h[0][edge_j])
        diff = torch.nn.functional.normalize(diff, dim=1)
        clust_i = density[edge_i].repeat(1, len(feat[0]))
        clust_j = density[edge_j].repeat(1, len(feat[0]))
        meta_info = torch.cat((feat, clust_i, clust_j), dim=1)
        m = self.late_meta_attn_fc_1(meta_info)
        m = F.relu(m)
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = self.late_meta_attn_fc_2(m)

        if len(local.shape) == 3:
            late_meta_scores = F.softmax(m, dim=2)
            meta_embedding = (late_meta_scores[:, :, 0].view(-1, late_meta_scores.shape[1], 1) * glob) + (late_meta_scores[:, :, 1].view(-1, late_meta_scores.shape[1], 1) * local)# + (late_meta_scores[:, :, 2].view(-1, late_meta_scores.shape[1], 1) * diff)
        else:
            if self.weights == 'meta':
                late_meta_scores = F.softmax(m, dim=1)
            else:
                late_meta_scores = torch.zeros(F.softmax(m, dim=1).shape).to(self.device)
            
                if self.weights == 'global':
                    late_meta_scores[:, 0] = 1.0
                if self.weights == 'local':
                    late_meta_scores[:, 1] = 1.0
                if self.weights == 'diff':
                    late_meta_scores[:, 2] = 1.0

            score_sum = torch.sum(late_meta_scores, dim=0)
            self.meta_distributions[0] += score_sum[0] / late_meta_scores.shape[0]
            self.meta_distributions[1] += score_sum[1] / late_meta_scores.shape[0]
            self.meta_distributions[2] += score_sum[2] / late_meta_scores.shape[0]
            self.meta_count += 1
            
            meta_embedding = (late_meta_scores[:, 0].view(-1, 1) * glob) + (late_meta_scores[:, 1].view(-1, 1) * local) + (late_meta_scores[:, 2].view(-1, 1) * diff)
        
        x = meta_embedding

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        score = torch.sigmoid(x)
        return score, torch.sigmoid(self.add_edge(score))

