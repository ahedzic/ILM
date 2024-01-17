import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import skfuzzy as fuzz

class ACCSLPModel(nn.Module):
    def __init__(self, config, gpu_id):
        super(ACCSLPModel, self).__init__()
        self.gpu_id = gpu_id
        self.config = config
        self.n = config['model_params']['max_nodes']
        self.rank = config['model_params']['rank']
        self.alpha = config['model_params']['alpha']
        self.beta = config['model_params']['beta']
        self.W = torch.rand(self.n, self.rank).to('cuda:' + self.gpu_id)
        self.H = torch.rand(self.rank, self.n).to('cuda:' + self.gpu_id)
        self.U = torch.rand(self.n, self.rank).to('cuda:' + self.gpu_id)
        self.V = torch.rand(self.rank, self.n).to('cuda:' + self.gpu_id)

    def forward(self, graph, inference=False):
        # S = # nodes x # nodes
        S = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)
        edges = graph.edges()

        # Fill out padded S with adj matrix values
        for i in range(graph.number_of_edges()):
            S[edges[0][i].item()][edges[1][i].item()] = 1
            S[edges[1][i].item()][edges[0][i].item()] = 1

        # Use cosine similarity to determine attribute similarity matrix for nodes features
        node_feats = torch.zeros((self.n, self.config['node_feature_size']))

        for i in range(len(graph.ndata['feat'])):
            node_feats[i] = graph.ndata['feat'][i]

        Z = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)

        for i in range(self.n):
            for j in range(self.n):
                dual_sum = 0.0
                first_sum = 0.0
                second_sum = 0.0

                for k in range(self.config['node_feature_size']):
                    dual_sum += node_feats[i][k] * node_feats[j][k]
                    first_sum += node_feats[i][k] * node_feats[i][k]
                    second_sum += node_feats[j][k] * node_feats[j][k]

                if (math.sqrt(first_sum * second_sum) > 0.0):
                    Z[i][j] = dual_sum / math.sqrt(first_sum * second_sum)

        # use Fuzzy C-means clustering to create a community membership matrix based off node attributes
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(torch.transpose(node_feats, 0, 1), self.config['model_params']['groups'], 2, error=0.005, maxiter=1000)
        node_membership = np.argmax(u, axis=0)
        X = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)

        for i in range(len(node_membership)):
            for j in range(len(node_membership)):
                if node_membership[i] == node_membership[j]:
                    X[i][j] = 1.0

        # Calculate current prime matrices
        S_prime = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)
        Z_prime = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)
        X_prime = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)
        e = 1.0e-10

        #print("U", self.U)
        #print("H", self.H)
        #print("V", self.V)
        #print("W", self.W)

        for l in range(self.n):
            for j in range(self.n):
                sum_uh = e
                sum_wh = e
                sum_uv = e

                for k in range(self.rank):
                    sum_uh += self.U[l][k] * self.H[k][j]
                    sum_wh += self.W[l][k] * self.H[k][j]
                    sum_uv += self.U[l][k] * self.V[k][j]

                S_prime[l][j] = sum_uh
                Z_prime[l][j] = sum_uv
                X_prime[l][j] = sum_wh

        if inference:
            S_part = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)
            X_part = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)
            Z_part = torch.zeros((self.n, self.n)).to('cuda:' + self.gpu_id)

            for i in range(self.n):
                for j in range(self.n):
                    if (S[i][j] == 0.0) or (S_prime[i][j] == 0.0):
                        S_part[i][j] = S_prime[i][j] - S[i][j]
                    else:
                        S_part[i][j] = S[i][j] * torch.log(S[i][j] / S_prime[i][j]) + S_prime[i][j] - S[i][j]

                    if (X[i][j] == 0.0) or (X_prime[i][j] == 0.0):
                        X_part[i][j] = X_prime[i][j] - X[i][j]
                    else:
                        X_part[i][j] = X[i][j] * torch.log(X[i][j] / X_prime[i][j]) + X_prime[i][j] - X[i][j]

                    if (Z[i][j] == 0.0) or (Z_prime[i][j] == 0.0):
                        Z_part[i][j] = Z_prime[i][j] - Z[i][j]
                    else:
                        Z_part[i][j] = Z[i][j] * torch.log(Z[i][j] / Z_prime[i][j]) + Z_prime[i][j] - Z[i][j]

            S_predict = S_part + self.alpha * X_part + self.beta * Z_part

            return S_predict
        else:
            for l in range(self.n):
                for k in range(self.rank):
                    sum_sh = e
                    sum_zv = e
                    sum_h = e
                    sum_v = e

                    for j in range(self.n):
                        sum_sh += S[l][j] * self.H[k][j] / S_prime[l][j]
                        sum_zv += Z[l][j] * self.V[k][j] / Z_prime[l][j]
                        sum_h += self.H[k][j]
                        sum_v += self.V[k][j]

                    self.U[l][k] = self.U[l][k] * (sum_sh + self.beta * sum_zv) / (sum_h + self.beta * sum_v)

            for k in range(self.rank):
                for j in range(self.n):
                    sum_su = e
                    sum_xw = e
                    sum_u = e
                    sum_w = e

                    for l in range(self.n):
                        sum_su += S[l][j] * self.U[l][k] / S_prime[l][j]
                        sum_xw += X[l][j] * self.W[l][k] / X_prime[l][j]
                        sum_u += self.U[l][k]
                        sum_w += self.W[l][k]

                    self.H[k][j] = self.H[k][j] * (sum_su + self.alpha * sum_xw) / (sum_u + self.alpha * sum_w)

            for i in range(self.n):
                for k in range(self.rank):
                    sum_xh = e
                    sum_h = e

                    for j in range(self.n):
                        sum_xh += X[i][j] * self.H[k][j] / X_prime[i][j]
                        sum_h += self.H[k][j]

                    self.W[i][k] = self.W[i][k] * sum_xh / sum_h

            for k in range(self.rank):
                for m in range(self.n):
                    sum_zu = e
                    sum_u = e

                    for l in range(self.n):
                        sum_zu += Z[l][m] * self.U[l][k] / Z_prime[l][m]
                        sum_u += self.U[l][k]

                    self.V[k][m] = self.V[k][m] * sum_zu / sum_u


