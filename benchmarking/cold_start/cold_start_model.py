import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import skfuzzy as fuzz
import time

class Leroy(torch.nn.Module):
    def __init__(self, device_id):
        super(Leroy, self).__init__()
        self.gpu_id = device_id

    def forward(self, graph, inference=False):
        node_groups = graph['node_groups'].to(self.gpu_id)
        groups_size = graph['groups_size'].to(self.gpu_id)

        # Calculate scores for node pairs
        len_groups = torch.sum(node_groups, dim=1).to(self.gpu_id)
        # Initialize pair scores with prod feature
        pair_scores = torch.matmul(len_groups.view(-1, 1), torch.transpose(len_groups.view(-1, 1), 1, 0)).to(self.gpu_id)
        coef_groups_size = torch.pow(torch.log(groups_size), -1.0).to(self.gpu_id)
        # Multiply prod feature by ad_ad_s feature
        for i in range(len(node_groups)):
            for j in range(len(node_groups)):
                intersect_groups = torch.where((node_groups[i] > 0.0) & (node_groups[i] == node_groups[j]), 1.0, 0.0).nonzero()
                pair_scores[i, j] *= torch.sum(coef_groups_size[intersect_groups].flatten())

        # Determine node_neighbors
        node_neighbors = torch.where(pair_scores > 0.0, 1.0, 0.0).to(self.gpu_id)
        # Convert pair scores to probabilities
        max_log = math.log(pair_scores.max()  + 1.0)
        pair_scores = torch.where(pair_scores > -1.0, torch.log(pair_scores + 1.0) / max_log, 0.0).to(self.gpu_id)
        # Calculate common neighbors
        common_neighbors = torch.zeros(pair_scores.shape).to(self.gpu_id)

        for i in range(len(node_groups)):
            for j in range(len(node_groups)):
                intersect_common_neighbors = torch.where((node_neighbors[i] > 0.0) & (node_neighbors[i] == node_neighbors[j]), 1.0, 0.0).nonzero()
                common_neighbors[i, j] += torch.dot(pair_scores[i, intersect_common_neighbors].view(-1), pair_scores[j, intersect_common_neighbors].view(-1))

        common_neighbors = torch.nan_to_num(common_neighbors)

        return common_neighbors


class ACCSLP(torch.nn.Module):
    def __init__(self, max_nodes, rank, alpha, beta, groups, device_id):
        super(ACCSLP, self).__init__()
        self.gpu_id = device_id
        self.groups = groups
        self.n = max_nodes
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.W = torch.ones(self.n, self.rank).to(self.gpu_id)
        self.H = torch.ones(self.rank, self.n).to(self.gpu_id)
        self.U = torch.ones(self.n, self.rank).to(self.gpu_id)
        self.V = torch.ones(self.rank, self.n).to(self.gpu_id)

    def forward(self, data, inference=False):
        if inference:
            return torch.matmul(self.U, self.H)
        else:
            S = data['adj'].to_dense().to(self.gpu_id)
            Z = data['attribute_matrix'].to(self.gpu_id)
            X = data['community_membership'].to(self.gpu_id)
            S_pred = torch.matmul(self.U, self.H)
            X_pred = torch.matmul(self.W, self.H)
            Z_pred = torch.matmul(self.U, self.V)
            S_obj = S_pred - S * torch.log(S_pred)
            X_obj = X_pred- X * torch.log(X_pred)
            Z_obj = Z_pred- Z * torch.log(Z_pred)
            diff = S_obj.sum() + self.alpha * X_obj.sum() + self.beta * Z_obj.sum()
            S_div = torch.div(S, S_pred)
            X_div = torch.div(X, X_pred)
            Z_div = torch.div(Z, Z_pred)

            while (diff > 0.0):
                U_coeff = torch.zeros(self.U.shape).to(self.gpu_id)
                H_coeff = torch.zeros(self.H.shape).to(self.gpu_id)
                W_coeff = torch.zeros(self.W.shape).to(self.gpu_id)
                V_coeff = torch.zeros(self.V.shape).to(self.gpu_id)

                for l in range(self.n):
                    for j in range(self.n):
                        for k in range(self.rank):
                            U_coeff[l,k] += (S_div[l,j] * self.H[k,j] + self.beta * Z_div[l,j] * self.V[k,j]) / (self.H[k,j] + self.beta * self.V[k,j])

                self.U = self.U * U_coeff
                S_pred = torch.matmul(self.U, self.H)
                Z_pred = torch.matmul(self.U, self.V)
                S_div = torch.div(S, S_pred)
                Z_div = torch.div(Z, Z_pred)

                for l in range(self.n):
                    for j in range(self.n):
                        for k in range(self.rank):
                            H_coeff[k,j] += (S_div[l,j] * self.U[l,k] + self.alpha * X_div[l,j] * self.W[l,k]) / (self.U[l,k] + self.alpha * self.W[l,k])

                self.H = self.H * H_coeff
                S_pred = torch.matmul(self.U, self.H)
                X_pred = torch.matmul(self.W, self.H)
                S_div = torch.div(S, S_pred)
                X_div = torch.div(X, X_pred)

                for l in range(self.n):
                    for j in range(self.n):
                        for k in range(self.rank):
                            W_coeff[l,k] += (X_div[l,j] * self.H[k,j]) / self.H[k,j]

                self.W = self.W * W_coeff
                X_pred = torch.matmul(self.W, self.H)
                X_div = torch.div(X, X_pred)

                for l in range(self.n):
                    for j in range(self.n):
                        for k in range(self.rank):
                            V_coeff[k,j] += (Z_div[l,j] * self.U[l,k]) / self.U[l,k]

                self.V = self.V * V_coeff
                Z_pred = torch.matmul(self.U, self.V)
                Z_div = torch.div(Z, Z_pred)
                
                S_obj = S_pred - S * torch.log(S_pred)
                X_obj = X_pred- X * torch.log(X_pred)
                Z_obj = Z_pred- Z * torch.log(Z_pred)
                O = S_obj.sum() + self.alpha * X_obj.sum() + self.beta * Z_obj.sum()
                #print("Current O", O)
                diff = diff - O

            #print("FINISHED CONVERGING")

            return O
            
    
'''class CSLP_LFR(torch.nn.Module):
    def __init__(self, max_nodes, rank, alpha, beta, groups, device_id):
        super(CSLP_LFR, self).__init__()
        self.gpu_id = device_id
        self.groups = groups
        self.n = max_nodes
        self.l = config['model_params']['latent_size']
        self.feat_size = config['node_feature_size']
        self.lr = config['model_params']['learning_rate']
        # Initialize random Z, omega, and phi
        self.Z = torch.zero(self.n, self.l).uniform_(0, 1)
        self.mu = torch.zero(self.n, self.l).uniform_(0, 1)
        self.omega = torch.zero(self.l, 2, 2).uniform_(0, 1)
        self.phi = torch.zero(self.n, self.l).uniform_(0, 1)
        self.wl = torch.zero(self.l, self.feat_size).uniform_(0, 1)

    def e_step(self, S):
        phi_new = copy.deepcopy(self.phi)

        for i in S:
            delta_LQ_phi = 
            delta_MI_phi = 0

    def m_step(self, S):
        for i in range(self.l):
            pass

        for 

    def forward(self, x, adj_t, inference=False, logistic=False):
        S = torch.zeros((self.n, self.n))
        edges = graph.edges()
        dense = adj_t.to_dense()

        # Fill out padded S with adj matrix values
        for i in range(len(dense)):
            S[dense[0][i].item()][dense[1][i].item()] = 1
            S[dense[1][i].item()][dense[0][i].item()] = 1

        if not inference:
            if not logistic:
                # Learn latent-feature representation Z and the link-affinity matrix set omega
                e_step(S)
                m_step(S)
            else:
                existing_z = []
                existing_a = []
                non_existing_a = []

                for node in graph.ndata['feat']:
                    if (len(graph.in_edges(node, form='eid')) > 0) or (len(graph.out_edges(node, form='eid')) > 0):
                        existing_a.append(node)
                        existing_z.append()//TODO
                    else:
                        non_existing_a.append(node)

                for i in range(len(existing_z)):
                    z = existing_z[i]
                    a = existing_a[i]
                    z_pred = torch.zero(self.l)
                    deltas = torch.zero(self.l)

                    for j in range(len(self.wl)):
                        y_sum = 0.0
                        z_pred = 0.0

                        for k in range(len(self.wl[j])):
                            y_sum += self.wl[j][k] * a[k]

                        y = 1.0/(1.0 + math.exp(-1.0 * y_sum))

                        if y >= 0.5:
                            z_pred[j] = 1.0

                        deltas[j] = (z_pred[j] - y) * a

                    for j in range(len(self.wl)):
                        self.wl[j] = self.wl[j] + self.lr * deltas[j]
        else:
            pass'''
    