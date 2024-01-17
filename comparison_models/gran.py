from . import graph_model
from . import gran_model
from . import gran_data
from . import data_parallel
from . import train_helper
import copy
import dgl
import os.path
import pickle
import random
import torch.optim as optim
import torch as th
import networkx as nx
import numpy as np
import time

class GRAN(graph_model.BaseGraphModel):
    def __init__(self, model_params, dataset, results, gpu_id):
        super(GRAN, self).__init__(model_params, dataset, results, gpu_id)

    def initialize_model(self):
        node_feature_size = self.model_params['node_feature_size']
        num_edge_types = self.model_params['num_edge_types']
        self.model_params['model_params']['device'] = 'cuda:' + self.gpu_id
        self.model = gran_model.GRANMixtureBernoulli(self.model_params['model_params'])
        self.model = data_parallel.DataParallel(self.model, ['cuda:' + self.gpu_id]).to('cuda:' + self.gpu_id)
        self.model_file = self.model_params['model'] + self.model_params['dataset_name'] + '_model'
        self.edge_loss_fn = th.nn.CrossEntropyLoss()

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.model_params['model_params']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.model_params['model_params']['lr'],
                momentum=self.model_params['model_params']['momentum'],
                weight_decay=self.model_params['model_params']['wd'])
        elif self.model_params['model_params']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(params, lr=self.model_params['model_params']['lr'], weight_decay=self.model_params['model_params']['wd'])
        else:
            raise ValueError("Non-supported optimizer!")

        self.early_stop = train_helper.EarlyStopper([0.0], win_size=100, is_decrease=False)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.model_params['model_params']['lr_decay_epoch'],
            gamma=self.model_params['model_params']['lr_decay'])

        self.optimizer.zero_grad()

        if not self.model_params['fresh_model']:
            if os.path.isfile(self.model_file):
                train_helper.load_model(
                    self.model.module,
                    self.model_file,
                    self.gpu_id,
                    optimizer=optimizer,
                    scheduler=lr_scheduler)

    def prepare_graph(self, graph_dict):
        if self.model_params["edge_percentage"] > 0.0:
            new_graph_dict = {}
            current_graph = copy.deepcopy(graph_dict['expected'])
            full_graph = copy.deepcopy(graph_dict['full'])
            left_edges = int(self.model_params["edge_percentage"] * (current_graph.number_of_edges()) + 1)

            if (current_graph.number_of_edges() - left_edges) == 0:
                left_edges -= 1
                        
            while current_graph.number_of_edges() > left_edges:
                index = random.randint(0, (current_graph.number_of_edges() - 1))
                current_graph = dgl.remove_edges(current_graph, [index])
                            
            edges = current_graph.edges(form='all')
            u_edges = edges[0]
            v_edges = edges[1]

            for y in range(len(u_edges)):
                u = u_edges[y]
                v = v_edges[y]
                full_edges = full_graph.edges(form='all')

                for y in range(len(full_edges[0])):
                    if full_edges[0][y] == u and full_edges[1][y] == v:
                        full_graph = dgl.remove_edges(full_graph, full_edges[2][y])

            new_graph_dict['empty'] = current_graph
            new_graph_dict['full'] = full_graph
            new_graph_dict['expected'] = graph_dict['expected']
            graph_dict = new_graph_dict
        else:
            if graph_dict['empty'].number_of_edges():
                graph_dict['empty'] = dgl.remove_edges(graph_dict['empty'], graph_dict['empty'].edges(form='eid'))

        return graph_dict

    def prepare_graphs(self, dataset):
        graphs = []
        self.true_graphs = []

        for graph_file in dataset:
            with open(graph_file, 'rb') as graph_dict_file:
                graph_dict = pickle.load(graph_dict_file)
                graph_dict = self.prepare_graph(graph_dict)
                true_graph = graph_dict['expected']
                empty_graph = graph_dict['empty']

                if (true_graph.number_of_nodes() <= 25) and (empty_graph.number_of_nodes() <= 25):
                    nx_graph = true_graph.to_networkx(node_attrs=['feat'], edge_attrs=['z'])
                    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
                    self.true_graphs.append(nx_graph)
                    empty_nx_graph = empty_graph.to_networkx(node_attrs=['feat'], edge_attrs=['z'])
                    graphs.append(empty_nx_graph)

        return graphs

    def train_model(self, train_dataset):
        epochs = self.model_params['epochs']
        graphs = self.prepare_graphs(train_dataset)
        num_nodes_pmf = np.bincount([len(gg.nodes) for gg in graphs])[1:]   
        num_nodes_pmf = num_nodes_pmf / num_nodes_pmf.sum()
        train_dataset = gran_data.GRANData(self.model_params['model_params'], graphs, self.true_graphs, tag='train')
        train_loader = th.utils.data.DataLoader(
            train_dataset,
            batch_size=self.model_params['model_params']['batch_size'],
            shuffle=False,
            num_workers=self.model_params['model_params']['num_workers'],
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        start_time = time.time()
        for _ in range(epochs):
            print("TRAINING EPOCH", _, "FOR MODEL", self.model_params['model'])
            # Training Loop
            self.model.train()
            self.lr_scheduler.step()
            train_iterator = train_loader.__iter__()
            iter_count = 0 
            avg_train_loss = .0

            for inner_iter in range(len(train_loader)):
                self.optimizer.zero_grad()

                batch_data = []
                data = train_iterator.next()
                batch_data.append(data)  
                iter_count += 1
                added = False

                for ff in range(self.model_params['model_params']['num_fwd_pass']):
                    batch_fwd = []
                                    
                    for dd, gpu_id in enumerate(['cuda:' + self.gpu_id]):
                        data = {}
                        data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
                        data['adj_true'] = batch_data[dd][ff]['adj_true'].pin_memory().to(gpu_id, non_blocking=True)
                        data['node_feat'] = batch_data[dd][ff]['node_feat'].pin_memory().to(gpu_id, non_blocking=True)       
                        data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
                        data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
                        data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
                        data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
                        data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
                        data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
                        data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
                        data['is_sampling'] = False
                        data['batch_size'] = self.model_params['model_params']['batch_size']
                        data['num_nodes_pmf'] = num_nodes_pmf
                        batch_fwd.append((data,))

                    if batch_fwd:
                        if data['adj_true'][:, 0, :, :].shape[1] == 25:
                            adj = self.model(*batch_fwd)
                            train_loss = self.edge_loss_fn(adj, data['adj_true'][:, 0, :, :])         
                            avg_train_loss += train_loss    
                            train_loss.backward()
                            added = True        

                # clip_grad_norm_(model.parameters(), 5.0e-0)
                if added:
                    self.optimizer.step()
                    #avg_train_loss /= float(self.model_params['model_params']['num_fwd_pass'])
                                    
                    # reduce
                    train_loss = float(train_loss.data.cpu().numpy())
                    #print("NLL Loss @ epoch {:04d} iteration {:08d} = {} {}".format(_ + 1, iter_count, train_loss, avg_train_loss / iter_count))

            train_helper.snapshot(self.model.module, self.optimizer, self.model_params['model_params'], _ + 1, scheduler=self.lr_scheduler)
        print("Finished training in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])

    def test_model(self, test_dataset):
        graph_results = []
        total_edges = { 1: 0}
        edges_per_iteration = 1
        graphs = self.prepare_graphs(test_dataset)
        num_nodes_pmf = np.bincount([len(gg.nodes) for gg in graphs])[1:]   
        num_nodes_pmf = num_nodes_pmf / num_nodes_pmf.sum()
        test_dataset = gran_data.GRANData(self.model_params['model_params'], graphs, self.true_graphs, tag='test')
        train_loader = th.utils.data.DataLoader(
            test_dataset,
            batch_size=self.model_params['model_params']['batch_size'],
            shuffle=False,
            num_workers=self.model_params['model_params']['num_workers'],
            collate_fn=test_dataset.collate_fn,
            drop_last=False)
        iter_count = 0 
        start_time = time.time()
        for _ in range(1):
            print("TESTING EPOCH", _, "FOR MODEL", self.model_params['model'])
            # Training Loop
            self.model.train()
            self.lr_scheduler.step()
            train_iterator = train_loader.__iter__()

            for inner_iter in range(len(train_loader)):
                self.optimizer.zero_grad()
                #print("TESTING ON GRAPH", iter_count)
                batch_data = []
                data = train_iterator.next()
                batch_data.append(data)  
                avg_train_loss = .0
                iter_count += 1

                for ff in range(self.model_params['model_params']['num_fwd_pass']):
                    batch_fwd = []
                                    
                    for dd, gpu_id in enumerate(['cuda:' + self.gpu_id]):
                        data = {}
                        data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)   
                        data['adj_true'] = batch_data[dd][ff]['adj_true'].pin_memory().to(gpu_id, non_blocking=True)
                        data['node_feat'] = batch_data[dd][ff]['node_feat'].pin_memory().to(gpu_id, non_blocking=True)       
                        data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
                        data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
                        data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
                        data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
                        data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
                        data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
                        data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
                        data['is_sampling'] = True
                        data['batch_size'] = self.model_params['model_params']['batch_size']
                        data['num_nodes_pmf'] = num_nodes_pmf
                        batch_fwd.append((data,))

                    if batch_fwd:
                        if data['adj_true'][:, 0, :, :].shape[1] == 25:
                            adj = self.model(*batch_fwd)
                            total = 0

                            for b in range(len(adj)):
                                for i in range(len(adj[b])):
                                    for j in range(len(adj[b])):
                                        if data['adj_true'][b][0][i][j] > 0:
                                            total_edges[1] += 1
                                            total += 1

                            for b in range(len(adj)):
                                graph_result = []

                                for i in range(len(adj[b])):
                                    for j in range(len(adj[b])):
                                        pred = -1
                                        label = -1

                                        if adj[b][i][j] > 0:
                                            pred = 1
                                        if data['adj_true'][b][0][i][j] > 0:
                                            label = 1
                                            
                                        graph_result.append([pred, label, total])

                                graph_results.append(graph_result)
        print("Finished testing in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])
        self.results.add_metrics(self.model_params['model'], graph_results, total_edges)