
import torch
import torch.nn.functional as F

import torch.nn as nn

class meta_score(torch.nn.Module):
    def __init__(self, feat_channels, in_channels, hidden_channels, out_channels, edge_types, num_layers,
                 dropout, weights, device):
        super(meta_score, self).__init__()

        self.device = device
        self.weights = weights
        self.late_meta_attn_fc_1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.late_meta_attn_fc_2 = nn.Linear(hidden_channels, 9)
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

        self.edge_lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.edge_lins.append(torch.nn.Linear(in_channels, edge_types))
        else:
            self.edge_lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.edge_lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.edge_lins.append(torch.nn.Linear(hidden_channels, edge_types))

        self.dropout = dropout
        self.reset_meta_distributions()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.late_meta_attn_fc_1.weight, gain=gain)
        nn.init.xavier_uniform_(self.late_meta_attn_fc_2.weight, gain=gain)
        nn.init.xavier_uniform_(self.feat_layer.weight, gain=gain)
        nn.init.xavier_uniform_(self.add_edge.weight, gain=gain)

        for lin in self.lins:
            lin.reset_parameters()

        for lin in self.edge_lins:
            lin.reset_parameters()

    def reset_meta_distributions(self):
        self.meta_distributions = torch.zeros(9).to(self.device)
        self.meta_count = 0

    def get_meta_distributions(self):
        return self.meta_distributions / self.meta_count

    def forward(self, x, h, h_empty, edge_i, edge_j, density=None):
        feat = self.feat_layer(x[edge_i] * x[edge_j])
        feat = torch.nn.functional.normalize(feat, dim=1)
        glob_0 = torch.sum(h_empty[0], dim=0)
        glob_0 = torch.nn.functional.normalize(glob_0.view(-1, glob_0.shape[0]))
        glob_0 = glob_0.repeat(len(edge_i), 1)
        glob_1 = torch.sum(h[0], dim=0)
        glob_1 = torch.nn.functional.normalize(glob_1.view(-1, glob_1.shape[0]))
        glob_1 = glob_1.repeat(len(edge_i), 1)
        glob_2 = torch.sum(h[1], dim=0)
        glob_2 = torch.nn.functional.normalize(glob_2.view(-1, glob_2.shape[0]))
        glob_2 = glob_2.repeat(len(edge_i), 1)
        sim_0 = h_empty[0][edge_i] + h_empty[0][edge_j]
        sim_0 = torch.nn.functional.normalize(sim_0, dim=1)
        sim_1 = h[0][edge_i] + h[0][edge_j]
        sim_1 = torch.nn.functional.normalize(sim_1, dim=1)
        sim_2 = h[1][edge_i] + h[1][edge_j]
        sim_2 = torch.nn.functional.normalize(sim_2, dim=1)
        diff_0 = abs(h_empty[0][edge_i] - h_empty[0][edge_j])
        diff_0 = torch.nn.functional.normalize(diff_0, dim=1)
        diff_1 = abs(h[0][edge_i] - h[0][edge_j])
        diff_1 = torch.nn.functional.normalize(diff_1, dim=1)
        diff_2 = abs(h[1][edge_i] - h[1][edge_j])
        diff_2 = torch.nn.functional.normalize(diff_2, dim=1)
        clust_i = density[edge_i].repeat(1, len(feat[0]))
        clust_j = density[edge_j].repeat(1, len(feat[0]))
        meta_info = torch.cat((feat, clust_i, clust_j), dim=1)
        m = self.late_meta_attn_fc_1(meta_info)
        m = F.relu(m)
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = self.late_meta_attn_fc_2(m)

        if len(sim_2.shape) == 3:
            late_meta_scores = F.softmax(m, dim=2)
            meta_embedding = (late_meta_scores[:, :, 0].view(-1, late_meta_scores.shape[1], 1) * glob) + (late_meta_scores[:, :, 1].view(-1, late_meta_scores.shape[1], 1) * local)# + (late_meta_scores[:, :, 2].view(-1, late_meta_scores.shape[1], 1) * diff)
        else:
            if self.weights == 'meta':
                late_meta_scores = F.softmax(m, dim=1)
            else:
                late_meta_scores = torch.zeros(F.softmax(m, dim=1).shape).to(self.device)
            
                if self.weights == 'global0':
                    late_meta_scores[:, 0] = 1.0
                if self.weights == 'global1':
                    late_meta_scores[:, 1] = 1.0
                if self.weights == 'global2':
                    late_meta_scores[:, 2] = 1.0
                if self.weights == 'sim0':
                    late_meta_scores[:, 3] = 1.0
                if self.weights == 'sim1':
                    late_meta_scores[:, 4] = 1.0
                if self.weights == 'sim2':
                    late_meta_scores[:, 5] = 1.0
                if self.weights == 'diff0':
                    late_meta_scores[:, 6] = 1.0
                if self.weights == 'diff1':
                    late_meta_scores[:, 7] = 1.0
                if self.weights == 'diff2':
                    late_meta_scores[:, 8] = 1.0

            score_sum = torch.sum(late_meta_scores, dim=0)
            self.meta_distributions[0] += score_sum[0].item() / late_meta_scores.shape[0]
            self.meta_distributions[1] += score_sum[1].item() / late_meta_scores.shape[0]
            self.meta_distributions[2] += score_sum[2].item() / late_meta_scores.shape[0]
            self.meta_distributions[3] += score_sum[3].item() / late_meta_scores.shape[0]
            self.meta_distributions[4] += score_sum[4].item() / late_meta_scores.shape[0]
            self.meta_distributions[5] += score_sum[5].item() / late_meta_scores.shape[0]
            self.meta_distributions[6] += score_sum[6].item() / late_meta_scores.shape[0]
            self.meta_distributions[7] += score_sum[7].item() / late_meta_scores.shape[0]
            self.meta_distributions[8] += score_sum[8].item() / late_meta_scores.shape[0]
            self.meta_count += 1
            
            meta_embedding = (late_meta_scores[:, 0].view(-1, 1) * glob_0) + (late_meta_scores[:, 1].view(-1, 1) * glob_1) + (late_meta_scores[:, 2].view(-1, 1) * glob_2) + (late_meta_scores[:, 3].view(-1, 1) * sim_0) + (late_meta_scores[:, 4].view(-1, 1) * sim_1) + (late_meta_scores[:, 5].view(-1, 1) * sim_2) + (late_meta_scores[:, 6].view(-1, 1) * diff_0) + (late_meta_scores[:, 7].view(-1, 1) * diff_1) + (late_meta_scores[:, 8].view(-1, 1) * diff_2)
        
        x = meta_embedding
        e_type = meta_embedding

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        score = torch.sigmoid(x)

        for lin in self.edge_lins[:-1]:
            e_type = lin(e_type)
            e_type = F.relu(e_type)
            e_type = F.dropout(e_type, p=self.dropout, training=self.training)
        e_type = self.edge_lins[-1](e_type)
        type_score = torch.argmax(e_type, dim=1)

        return score, torch.sigmoid(self.add_edge(score)), type_score, e_type

