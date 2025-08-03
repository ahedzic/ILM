import dgl
from generate_dataset import generate_dataset

import numpy as np
import torch
import csv


class DataModel():
    def __init__(self, data, adj_selector='adj_multi'):
        super(DataModel, self).__init__()
        self.load_data(data, adj_selector)

    def load_data(self, data, adj_selector):
        self.adj = data[adj_selector]
        self.drug_num = self.adj.shape[0]
        self.drug_ids = data['drug_ids']

        features = [
            data["feature_dbp"],
            data["feature_structure"],
        ]
        self.load_features(features)
        self.interaction_num = self.adj.max().astype(int).item() + 1

    def load_features(self, features):
        self.view_num = len(features)
        self.features = []
        self.view_dims = []

        for feature in features:
            self.features.append(feature)
            self.view_dims.append(feature.shape[1])

        self.feature = self.features[0]

def main():
    data = np.load("drugbank_v5_stanfordnlp.npz", allow_pickle=True)['data'].item()
    dataset = DataModel(data, adj_selector='adj_multi')
    num_nodes = dataset.drug_num
    adj_matrix = torch.from_numpy(dataset.adj)
    src, dst = torch.nonzero(adj_matrix, as_tuple=True)
    edge_types = adj_matrix[src, dst]
    csmddi_graph = dgl.graph((adj_matrix.nonzero(as_tuple=True)))
    csmddi_graph.ndata['feat'] = torch.from_numpy(dataset.feature).float()
    csmddi_graph.edata['edge_types'] = edge_types.float()
    generate_dataset('csmddi', csmddi_graph, 'feat', num_nodes / 21, 10, [0.8, 0.1, 0.1], True)

if __name__ == '__main__':
    main()
