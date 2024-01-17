from . import graph_model
from . import accslp_model
from . import train_helper
import copy
import dgl
import os.path
import pickle
import random
import torch.optim as optim
import torch as th
import time

class ACCSLP(graph_model.BaseGraphModel):
    def __init__(self, model_params, dataset, results, gpu_id):
        super(ACCSLP, self).__init__(model_params, dataset, results, gpu_id)

    def initialize_model(self):
        self.model_file = self.model_params['model'] + self.model_params['dataset_name'] + '_model'
        self.model = accslp_model.ACCSLPModel(self.model_params, self.gpu_id).to('cuda:' + self.gpu_id)

        if not self.model_params['fresh_model']:
            if os.path.isfile(self.model_file):
                self.model = pickle.load(open(self.model_file, 'rb'))
    
    def prepare_graph(self, graph_dict):
        if self.model_params["random_edge_percentage"]:
            perc = [0.0, 0.25, 0.5, 0.75]
            self.model_params["edge_percentage"] = random.choice(perc)

        if self.model_params["edge_percentage"] > 0.0:
            new_graph_dict = {}
            current_graph = copy.deepcopy(graph_dict['expected'])
            full_graph = copy.deepcopy(graph_dict['full'])

            if self.model_params["edge_percentage"] < 1.0:
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

    def train_model(self, train_dataset):
        epochs = self.model_params['epochs']
        start_time = time.time()

        for _ in range(epochs):
            total = 0
            print("TRAINING EPOCH", _, "FOR MODEL", self.model_params['model'], flush=True)
            for graph_file in train_dataset:
                with open(graph_file, 'rb') as graph_dict_file:
                    total += 1
                    graph_dict = pickle.load(graph_dict_file)
                    graph_dict = self.prepare_graph(graph_dict)
                    true_graph = graph_dict['expected'].to('cuda:' + self.gpu_id)
                    self.model.forward(true_graph)
                    print("FINISHED GRAPH", total, "OUT OF", len(train_dataset), "IN EPOCH", _, "FOR MODEL", self.model_params['model'], flush=True)

            pickle.dump(self.model, open(self.model_file, 'wb'))
        print("Finished training in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])

    def test_model(self, test_dataset):
        graph_results = []
        total_edges = {}
        total = 0
        total_edges[1] = 0
        start_time = time.time()
        for graph_file in test_dataset:
            with open(graph_file, 'rb') as graph_dict_file:
                total += 1
                graph_dict = pickle.load(graph_dict_file)
                graph_dict = self.prepare_graph(graph_dict)
                empty_graph = graph_dict['empty']
                true_graph = graph_dict['expected']
                edge_labels = graph_dict['full'].edata['labels']
                single_total = 0

                '''for edge in edge_labels:
                    if sum(edge) > 0:
                        if 1 in total_edges.keys():
                            total_edges[1] += 1
                        else:
                            total_edges[1] = 1
                        single_total += 1'''

                prediction = self.model.forward(empty_graph, True)
                graph_result = [] # First values prediction, second value label
                true_adj = th.zeros((true_graph.number_of_nodes(), true_graph.number_of_nodes()))
                true_edges = true_graph.edges()

                single_total = 0

                for i in range(true_graph.number_of_edges()):
                    true_adj[true_edges[0][i].item()][true_edges[1][i].item()] = 1#list(true_graph.edata['z'][i]).index(max(true_graph.edata['z'][i])) + 1

                for i in range(len(true_adj)):
                    for j in range(len(true_adj)):
                        if i != j and true_adj[i][j] > 0:
                            single_total += 1
                            total_edges[1] += 1

                for i in range(len(true_adj)):
                    for j in range(len(true_adj)):
                            if true_adj[i][j] != 8:
                                if prediction[i][j] > self.model_params['edge_score_threshold']:
                                    if true_adj[i][j] > self.model_params['edge_score_threshold']:
                                        #graph_result.append([int(true_adj[i][j]) - 1, int(true_adj[i][j]) - 1, single_total])
                                        graph_result.append([1, 1, single_total])
                                    else:
                                        graph_result.append([1, -1, single_total])
                                else:
                                    if true_adj[i][j] > self.model_params['edge_score_threshold']:
                                        #graph_result.append([-1, int(true_adj[i][j]) - 1, single_total])
                                        graph_result.append([-1, 1, single_total])
                                    else:
                                        graph_result.append([-1, -1, single_total])

                graph_results.append(graph_result)
                #print("TESTING", total, "OUT OF", len(test_dataset), "FOR MODEL", self.model_params['model'])
                
        print("Finished testing in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])
        self.results.add_metrics(self.model_params['model'], graph_results, total_edges)

                