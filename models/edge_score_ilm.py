import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EdgeScore(nn.Module):
    def __init__(self, h_feats, num_edge_types, struct_input_size, node_feature_size, group_count, gpu_id):
        super().__init__()
        #self.W1 = nn.Linear(h_feats + 1 + 2 * group_count, int((h_feats + 1 + 2 * group_count) / 2))
        #self.W2 = nn.Linear(int((h_feats + 1 + 2 * group_count) / 2), num_edge_types)
        self.W1 = nn.Linear(h_feats + 2 * group_count, int((h_feats + 2 * group_count) / 2))
        self.W2 = nn.Linear(int((h_feats + 2 * group_count) / 2), num_edge_types)
        self.softmax = nn.Softmax(dim=1)
        #self.cosine_sim = nn.CosineSimilarity()
        self.group_1 = nn.Linear(struct_input_size, int(struct_input_size / 2))
        self.group_2 = nn.Linear(int(struct_input_size / 2), group_count)
        self.late_meta_attn_fc_1 = nn.Linear(2 * struct_input_size, struct_input_size)
        self.late_meta_attn_fc_2 = nn.Linear(struct_input_size, 3)
        #self.add_edge_1 = nn.Linear(h_feats + 1 + 2 * group_count, int((h_feats + 1 + 2 * group_count) / 2))
        #self.add_edge_2 = nn.Linear(int((h_feats + 1 + 2 * group_count) / 2), 1)
        self.add_edge_1 = nn.Linear(h_feats + 2 * group_count, int((h_feats + 2 * group_count) / 2))
        self.add_edge_2 = nn.Linear(int((h_feats + 2 * group_count) / 2), 1)
        self.gpu_id = gpu_id
        self.num_edge_types = num_edge_types
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(int((h_feats + 2 * group_count) / 2))
        self.batchnorm_add = nn.BatchNorm1d(int((h_feats + 2 * group_count) / 2))
        self.batchnorm_group = nn.BatchNorm1d(int(struct_input_size / 2))
        self.batchnorm2 = nn.BatchNorm1d(num_edge_types)
        self.batchnorm_add2 = nn.BatchNorm1d(1)
        self.batchnorm_group2 = nn.BatchNorm1d(group_count)
        self.batchnorm_meta = nn.BatchNorm1d(struct_input_size)
        self.batchnorm_meta2 = nn.BatchNorm1d(3)
        self.group_count = group_count

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W1.weight, gain=gain)
        nn.init.xavier_normal_(self.W2.weight, gain=gain)
        nn.init.xavier_normal_(self.late_meta_attn_fc_1.weight, gain=gain)
        nn.init.xavier_normal_(self.late_meta_attn_fc_2.weight, gain=gain)
        nn.init.xavier_normal_(self.add_edge_1.weight, gain=gain)
        nn.init.xavier_normal_(self.add_edge_2.weight, gain=gain)
        nn.init.xavier_normal_(self.group_1.weight, gain=gain)
        nn.init.xavier_normal_(self.group_2.weight, gain=gain)

    def apply_edges(self, edges):
        pairs = torch.cat([edges.src['az'], edges.dst['az']], 1)
        if torch.isnan(pairs).any():
            print("pairs is NaN", flush=True)
            return 0
        late_meta_scores = F.softmax(self.batchnorm_meta2(self.late_meta_attn_fc_2(F.leaky_relu(self.batchnorm_meta(self.late_meta_attn_fc_1(pairs))))), dim=1)
        if torch.isnan(late_meta_scores).any():
            print("late_meta_scores is NaN", flush=True)
            return 0
        self.graph_embedding_matrix = torch.zeros((len(edges.src['h']), self.graph_embedding.size(dim=0))).to('cuda:' + self.gpu_id)

        for i in range(len(self.graph_embedding_matrix)):
            local = edges.src['az'][i] + edges.dst['az'][i]
            local_norm = local.norm(p=2, dim=0, keepdim=True)

            if local_norm > 0.0:
                local = local.div(local_norm)

            diff = abs(edges.src['az'][i] - edges.dst['az'][i])
            diff_norm = diff.norm(p=2, dim=0, keepdim=True)

            if diff_norm > 0.0:
                diff = diff.div(diff_norm)

            #late_meta_scores[i][0] = 0.0
            self.weighted_embedding = self.graph_embedding.clone() * late_meta_scores[i][0] + local * late_meta_scores[i][1] + diff * late_meta_scores[i][2]
            #self.weighted_embedding = self.graph_embedding.clone() * late_meta_scores[i][0] + diff * late_meta_scores[i][1]
            norm = self.weighted_embedding.norm(p=2, dim=0, keepdim=True)

            if norm > 0.0:
                self.graph_embedding_matrix[i] = self.weighted_embedding.div(norm)
            else:
                self.graph_embedding_matrix[i] = self.weighted_embedding

        #h = torch.cat([edges.src['h'], edges.dst['h'], self.graph_embedding_matrix, self.cosine_sim(edges.src['az'], edges.dst['az']).view(-1, 1) * late_meta_scores[i][3], self.softmax(self.group_2(F.leaky_relu(self.group_1(edges.src['az'])))).view(-1, self.group_count) * late_meta_scores[i][4], self.softmax(self.group_2(F.leaky_relu(self.group_1(edges.dst['az'])))).view(-1, self.group_count) * late_meta_scores[i][4]], 1)
        h = torch.cat([edges.src['h'], edges.dst['h'], self.graph_embedding_matrix, self.softmax(self.batchnorm_group2(self.group_2(F.leaky_relu(self.batchnorm_group(self.group_1(edges.src['az'])))))).view(-1, self.group_count), self.softmax(self.batchnorm_group2(self.group_2(F.leaky_relu(self.batchnorm_group(self.group_1(edges.dst['az'])))))).view(-1, self.group_count)], 1)
        self.meta_scores = late_meta_scores
        
        if self.num_edge_types > 1:
            scores = self.softmax(self.batchnorm2(self.W2(F.leaky_relu(self.batchnorm(self.W1(h))))))
        else:
            scores = self.sigmoid(self.batchnorm2(self.W2(F.leaky_relu(self.batchnorm(self.W1(h))))))
            empty = torch.zeros(scores.shape[0]).to('cuda:' + str(scores.get_device()))

        if self.num_edge_types > 1:
            return {'score': scores, 'add_edge_score': self.sigmoid(self.batchnorm_add2(self.add_edge_2(F.leaky_relu(self.batchnorm_add(self.add_edge_1(h))))))}
        else:
            return {'score': scores, 'add_edge_score': empty}

    def forward(self, g, graph_embedding, node_embeddings):
        self.graph_embedding = graph_embedding

        with g.local_scope():
            # Struct attention scores
            g.ndata['az'] = node_embeddings
            g.ndata['h'] = g.ndata['feat']
            g.apply_edges(self.apply_edges)

            return g.edata['score'], self.meta_scores, g.edata['add_edge_score']
