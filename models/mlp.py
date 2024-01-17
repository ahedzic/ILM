from . import graph_model
from . import edge_score_mlp
import itertools
import os.path
import pickle
import torch as th
import time

class MLP(graph_model.BaseGraphModel):
    def __init__(self, model_params, dataset, results, gpu_id):
        super(MLP, self).__init__(model_params, dataset, results, gpu_id)

    def initialize_model(self):
        node_feature_size = self.model_params['node_feature_size']
        num_edge_types = self.model_params['num_edge_types']
        self.edge_model = edge_score_mlp.EdgeScore((2 * node_feature_size), num_edge_types).to('cuda:' + self.gpu_id)
        self.model_file = self.model_params['model'] + self.model_params['dataset_name'] + '_model'

        if not self.model_params['fresh_model']:
            if os.path.isfile(self.model_file):
                self.edge_model.load_state_dict(th.load(self.model_file))

        self.optimizer = th.optim.Adam(itertools.chain(self.edge_model.parameters()), lr=0.001)
        self.multi_edge_loss_fn = th.nn.CrossEntropyLoss()
        self.edge_loss_fn = th.nn.BCELoss()

    def train_model(self, train_dataset):
        epochs = self.model_params['epochs']
        total = 0
        start_time = time.time()

        for _ in range(epochs):
            print("TRAINING EPOCH", _, "FOR MODEL", self.model_params['model'])
            for graph_file in train_dataset:
                with open(graph_file, 'rb') as graph_dict_file:
                    total += 1
                    graph_dict = pickle.load(graph_dict_file)
                    graph_dict = self.prepare_graph(graph_dict)
                    empty_graph = graph_dict['empty'].to('cuda:' + self.gpu_id)
                    full_graph = graph_dict['full'].to('cuda:' + self.gpu_id)
                    edge_labels = graph_dict['full'].edata['labels'].to('cuda:' + self.gpu_id)
                    add_edge_labels = th.zeros((edge_labels.shape[0], 1)).to('cuda:' + self.gpu_id)

                    if self.model_params['num_edge_types'] > 1:
                        for i in range(len(edge_labels)):
                            if sum(edge_labels[i]) > 0.0:
                                add_edge_labels[i] = 1.0

                    if full_graph.number_of_edges() > 1:
                        edge_scores, add_edge_scores = self.edge_model(full_graph)

                        if self.model_params['num_edge_types'] > 1:
                            edge_loss = self.multi_edge_loss_fn(th.clamp(edge_scores, 0.0, 1.0), edge_labels)
                            add_edge_loss = self.edge_loss_fn(th.clamp(add_edge_scores, 0.0, 1.0).view(-1, 1), add_edge_labels.view(-1, 1))
                            total_loss = edge_loss + add_edge_loss
                        else:
                            total_loss = self.edge_loss_fn(edge_scores.view(-1, 1), edge_labels.view(-1, 1))

                        total_loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # clear graphs from GPU memory after use
                        del empty_graph
                        del full_graph
                        del edge_labels
                        del add_edge_labels
                        th.cuda.empty_cache()

                    #print("FINISHED GRAPH", total, "OUT OF", len(train_dataset), "IN EPOCH", _, "FOR MODEL", self.model_params['model'], flush=True)
            
            th.save(self.edge_model.state_dict(), self.model_file)
        print("Finished training in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])

    def test_model(self, test_dataset):
        graph_results = []
        total_edges = {}
        edges_per_iteration = 1
        start_time = time.time()

        for graph_file in test_dataset:
            with open(graph_file, 'rb') as graph_dict_file:
                graph_dict = pickle.load(graph_dict_file)
                graph_dict = self.prepare_graph(graph_dict)
                empty_graph = graph_dict['empty'].to('cuda:' + self.gpu_id)
                full_graph = graph_dict['full'].to('cuda:' + self.gpu_id)
                edge_labels = graph_dict['full'].edata['labels'].to('cuda:' + self.gpu_id)
                single_total = 0
                add_edge_labels = th.zeros((edge_labels.shape[0], 1))

                if self.model_params['num_edge_types'] > 1:
                    for i in range(len(edge_labels)):
                        if sum(edge_labels[i]) > 0.0:
                            add_edge_labels[i] = 1.0

                for edge in edge_labels:
                    for e in range(len(edge)):
                        if edge[e] > 0:
                            if 0 in total_edges.keys():
                                total_edges[0] += 1
                            else:
                                total_edges[0] = 1
                            #if e in total_edges.keys():
                            #    total_edges[e] += 1
                            #else:
                            #    total_edges[e] = 1
                            single_total += 1

                if full_graph.number_of_edges() > 1:
                    generated_edges = []
                    generated_edge_labels = []
                    generated_add_edges = []
                    generated_add_edge_labels = []

                    for e in range(edges_per_iteration):
                        edge_scores, add_edge_scores = self.edge_model(full_graph)

                        if self.model_params['num_edge_types'] > 1:
                            add_edge_scores.reshape(-1, 1)
                        else:
                            edge_scores = edge_scores.reshape(-1, 1)

                        edge_index = 0
                        feature_index = 0
                        edges = full_graph.edges()
                        edges_u = edges[0].to(th.int64)
                        edges_v = edges[1].to(th.int64)

                        for i in range(len(edge_scores)):
                            edge = edge_scores[i]

                            if self.model_params['num_edge_types'] > 1:
                                add_edge = add_edge_scores[i]
                                new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                edge_index = i
                                max_score = -1

                                for j in range(len(edge)):
                                    edge_type = edge[j]

                                    if edge_type > max_score:
                                        max_score = edge_type
                                        feature_index = j

                                if add_edge >= 0.5:
                                    new_edge_feature[0][feature_index] = 1.
                                    edge_u = edges_u[edge_index]
                                    edge_v = edges_v[edge_index]
                                                
                                    # add new edge
                                    empty_graph.add_edges(th.tensor([edge_u]).to('cuda:' + self.gpu_id), th.tensor([edge_v]).to('cuda:' + self.gpu_id), { 'z': new_edge_feature })
                                    generated_edges.append(edge_scores[edge_index])
                                    generated_edge_labels.append(edge_labels[edge_index])
                                    generated_add_edges.append(add_edge)
                                    generated_add_edge_labels.append(add_edge_labels[edge_index])
                            else:
                                if edge >= 0.5:
                                    new_edge_feature = th.tensor([[1.]]).to('cuda:' + self.gpu_id)
                                    edge_u = edges_u[edge_index]
                                    edge_v = edges_v[edge_index]
                                                
                                    # add new edge
                                    empty_graph.add_edges(th.tensor([edge_u]).to('cuda:' + self.gpu_id), th.tensor([edge_v]).to('cuda:' + self.gpu_id), { 'z': new_edge_feature })
                                    generated_edges.append(edge_scores[edge_index])
                                    generated_edge_labels.append(edge_labels[edge_index])
                                    generated_add_edges.append(edge_scores[edge_index])
                                    generated_add_edge_labels.append(edge_labels[edge_index])


                    if len(generated_edges) > 0:
                        generated_edges = th.stack(generated_edges).to('cuda:' + self.gpu_id)
                        generated_edge_labels = th.stack(generated_edge_labels).to('cuda:' + self.gpu_id)
                        graph_result = [] # First values prediction, second value label

                        for i in range(len(generated_edges)):
                            if self.model_params['num_edge_types'] > 1:
                                gen_edge = list(generated_edges[i])
                                gen_edge_label = list(generated_edge_labels[i])
                                max_edge = max(gen_edge)
                                label_sum = sum(gen_edge_label)

                                if label_sum < 1.0:
                                    graph_result.append([0, -1, single_total])
                                    #graph_result.append([gen_edge.index(max_edge), -1, single_total])
                                else:
                                    graph_result.append([0, 0, single_total])
                                    #graph_result.append([gen_edge.index(max_edge), gen_edge_label.index(max(gen_edge_label)), single_total])
                            else:
                                if generated_edges[i] >= 0.5:
                                    if generated_edge_labels[i] < 1.0:
                                        graph_result.append([0, -1, single_total])
                                    else:
                                        graph_result.append([0, 0, single_total])

                        graph_results.append(graph_result)
                    else:
                        #print(self.model_params['model'], "NO EDGES ADDED", graph_file)
                        graph_results.append([])

                    # clear graphs from GPU memory after use
                    del empty_graph
                    del full_graph
                    del edge_labels
                    del generated_edges
                    del generated_edge_labels
                    th.cuda.empty_cache()
        print("Finished testing in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"], len(test_dataset))

        self.results.add_metrics(self.model_params['model'], graph_results, total_edges)