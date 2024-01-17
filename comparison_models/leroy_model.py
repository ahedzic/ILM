import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import skfuzzy as fuzz

class LeroyModel(nn.Module):
    def __init__(self, config):
        super(LeroyModel, self).__init__()
        self.config = config
        self.thresh = config['model_params']['threshold']

    def get_groups(self, node_feats, group_thresh):
        groups = set()

        for i in range(len(node_feats)):
            if node_feats[i] >= group_thresh:
                groups.add(i)

        return groups

    def get_pair_score(self, u_groups, v_groups, groups_size):
        prod = len(u_groups) * len(v_groups)
        ad_ad_s = 0.0
        c_groups = u_groups.intersection(v_groups)

        for group in c_groups:
            ad_ad_s += 1.0 / math.log(groups_size[group])

        return prod * ad_ad_s

    def forward(self, graph):
        node_groups = []
        groups_size = {}
        pair_scores = {}
        max_score = -1.0
        node_neighbors = {}
        common_neighbors = {}
        S = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
        g_max = graph.ndata['feat'].max().item()
        g_min = graph.ndata['feat'].min().item()

        if g_max > 1.0:
            g_max = 1.0
        if g_min < -1.0:
            g_min = -1.0

        group_thresh = g_max - ((g_max - g_min) / 2)

        # Initialize groups size dict
        for i in range(graph.number_of_nodes()):
            node_neighbors[i] = set()

        for i in range(len(graph.ndata['feat'][0])):
            groups_size[i] = 0

        # Determine groups from node features
        for i in range(len(graph.ndata['feat'])):
            groups = self.get_groups(graph.ndata['feat'][i], group_thresh)

            for group in groups:
                groups_size[group] += 1

            node_groups.append(groups)

        # Calculate scores for node pairs
        for i in range(graph.number_of_nodes()):
            for j in range(graph.number_of_nodes()):
                if i != j:
                    pair_score = self.get_pair_score(node_groups[i], node_groups[j], groups_size)
                    pair_scores[(i, j)] = pair_score

                    if pair_score > max_score:
                        max_score = pair_score
                    if pair_score > 0.0:
                        node_neighbors[i].add(j)
                        node_neighbors[j].add(i)

        # Convert pair scores to probabilities
        for k in pair_scores.keys():
            if max_score > 0.0:
                pair_scores[k] = math.log(pair_scores[k] + 1) / math.log(max_score + 1)
            else:
                pair_scores[k] = 0.0

        # Calculate common neighbors
        for k in pair_scores.keys():
            u, v = k[0], k[1]
            u_neigh, v_neigh = node_neighbors[u], node_neighbors[v]
            shared = u_neigh.intersection(v_neigh)
            common_neighbors[(u, v)] = 0.0

            for n in shared:
                common_neighbors[(u, v)] += pair_scores[(u, n)] * pair_scores[(v, n)]

        # Add edges
        for k in common_neighbors.keys():
            if common_neighbors[k] > self.thresh:
                S[k[0]][k[1]] = 1
                S[k[1]][k[0]] = 1

        return S
