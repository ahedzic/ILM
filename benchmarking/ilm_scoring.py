
import torch
import torch.nn.functional as F

import torch.nn as nn

class ilm_score(torch.nn.Module):
    def __init__(self, feature_channels, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(ilm_score, self).__init__()

        self.softmax = nn.Softmax(dim=1)
        self.late_meta_attn_fc_1 = nn.Linear(feature_channels, hidden_channels)
        self.late_meta_attn_fc_2 = nn.Linear(hidden_channels, 2)
        input_size = in_channels#2 * feature_channels + in_channels
        self.feat_layer = nn.Linear(feature_channels, in_channels)

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(input_size, out_channels))
        else:
            self.lins.append(torch.nn.Linear(input_size, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.late_meta_attn_fc_1.weight, gain=gain)
        nn.init.xavier_normal_(self.late_meta_attn_fc_2.weight, gain=gain)

        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, l_x_i, l_x_j, g_x_i, g_x_j):
        raw_feat = x_i * x_j
        glob = g_x_i * g_x_j
        local = l_x_i * l_x_j
        x_i = self.feat_layer(x_i)
        x_j = self.feat_layer(x_j)
        feat = x_i * x_j
        #diff = abs(l_x_i - l_x_j)
        m = self.late_meta_attn_fc_1(raw_feat)
        m = F.relu(m)
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = self.late_meta_attn_fc_2(m)

        if len(local.shape) == 3:
            late_meta_scores = F.softmax(m, dim=2)
            meta_embedding = (late_meta_scores[:, :, 0].view(-1, late_meta_scores.shape[1], 1) * feat) + (late_meta_scores[:, :, 1].view(-1, late_meta_scores.shape[1], 1) * local)# + (late_meta_scores[:, :, 0].view(-1, 500, 1) * diff)
            #x = torch.cat([x_i, x_j, meta_embedding], 2)
        else:
            late_meta_scores = F.softmax(m, dim=1)
            meta_embedding = (late_meta_scores[:, 0].view(-1, 1) * feat) + (late_meta_scores[:, 1].view(-1, 1) * local)# + (late_meta_scores[:, 2].view(-1, 1) * diff)
            #x = torch.cat([x_i, x_j, meta_embedding], 1)
        x = meta_embedding
        scores = x

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return torch.sigmoid(x), scores

