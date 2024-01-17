import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HGATLayer(nn.Module):
    def __init__(self, in_dim, num_edge_types):
        super(HGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, in_dim, bias=False)
        self.edge_fc = nn.Linear(num_edge_types, num_edge_types, bias=False)
        self.relation_attn_fc = nn.Linear(2 * in_dim + num_edge_types, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.relation_attn_fc.weight, gain=gain)

    def relationship_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.data['z'], edges.dst['z']], dim=1)
        a = self.relation_attn_fc(z2)

        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'fz': edges.data['z'], 'd': edges.dst['z']}

    def reduce_relationship_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        head_edge_tail = torch.cat([nodes.mailbox['z'], nodes.mailbox['fz'], nodes.mailbox['d']], dim=2)
        h = torch.sum(alpha * head_edge_tail, dim=1)

        return {'h': h}

    def forward(self, g):
        with g.local_scope():
            self.g = g
            z = self.fc(self.g.ndata['z'])
            self.g.ndata['z'] = z
            ez = self.edge_fc(self.g.edata['z'])
            self.g.edata['z'] = ez
            self.g.apply_edges(self.relationship_attention)
            self.g.update_all(self.message_func, self.reduce_relationship_func)

            return self.g.ndata.pop('h')

class MultiHeadHGATLayer(nn.Module):
    def __init__(self, in_dim, num_heads, num_edge_types, merge='cat'):
        super(MultiHeadHGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(HGATLayer(in_dim, num_edge_types))
        self.merge = merge

    def forward(self, g):
        heads_out = [attn_head(g) for attn_head in self.heads]

        if self.merge == 'cat':
            return torch.cat(heads_out, dim=1)
        elif self.merge == 'average':
            return torch.mean(torch.stack(heads_out))

class HGAT(nn.Module):
    def __init__(self, in_dim, num_heads, num_edge_types, graph_dim):
        super(HGAT, self).__init__()
        self.layer1 = MultiHeadHGATLayer(in_dim, num_heads, num_edge_types, 'cat')
        self.layer2 = MultiHeadHGATLayer(((2 * in_dim) + num_edge_types) * num_heads, 1, num_edge_types, 'cat')
        self.graph_fc = nn.Linear((2 * ((2 * in_dim) + num_edge_types) * num_heads) + num_edge_types, int(graph_dim / 2), bias=False)
        self.graph_fc2 = nn.Linear(int(graph_dim / 2), graph_dim, bias=False)
        self.graph_dim = graph_dim
        self.graph_embedding = None
        self.graph_embedding_normalized = None

    def forward(self, g, gpu_id):
        g.ndata['z'] = g.ndata['feat']
        h = self.layer1(g)
        h = F.elu(h)
        g.ndata['z'] = h
        h = self.layer2(g)
        self.graph_embedding = torch.zeros(self.graph_dim).to('cuda:' + gpu_id)

        for node in h:
            node_embedding = self.graph_fc2(F.elu(self.graph_fc(node)))
            self.graph_embedding += node_embedding

        norm = self.graph_embedding.norm(p=2, dim=0, keepdim=True)
        self.graph_embedding_normalized = self.graph_embedding.div(norm)

        return self.graph_embedding_normalized