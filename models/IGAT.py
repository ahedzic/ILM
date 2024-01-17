from . import graph_model
from . import edge_score, HGAT
import copy
import dgl
import itertools
import os.path
import pickle
import torch as th
import time

class IGAT(graph_model.BaseGraphModel):
    def __init__(self, model_params, dataset, results, gpu_id):
        super(IGAT, self).__init__(model_params, dataset, results, gpu_id)

    def initialize_model(self):
        node_feature_size = self.model_params['node_feature_size']
        num_edge_types = self.model_params['num_edge_types']
        attention_heads = self.model_params['model_params']['attention_heads']
        self.graph_embedding_size = self.model_params['model_params']['graph_embedding_size']
        self.graph_model = HGAT.HGAT(node_feature_size, attention_heads, num_edge_types, self.graph_embedding_size).to('cuda:' + self.gpu_id)
        self.edge_model = edge_score.EdgeScore((2 * node_feature_size) + self.graph_embedding_size, num_edge_types, self.model_params['model_params']['graph_embedding_size'], self.gpu_id).to('cuda:' + self.gpu_id)
        self.model_graph_file = self.model_params['model'] + self.model_params['dataset_name'] + '_graph_model'
        self.model_edge_file = self.model_params['model'] + self.model_params['dataset_name'] + '_edge_model'

        if not self.model_params['fresh_model']:
            if os.path.isfile(self.model_graph_file):
                self.graph_model.load_state_dict(th.load(self.model_graph_file))
            if os.path.isfile(self.model_edge_file):
                self.edge_model.load_state_dict(th.load(self.model_edge_file))

        self.optimizer = th.optim.Adam(itertools.chain(self.edge_model.parameters(), self.graph_model.parameters()), lr=0.001)
        self.multi_edge_loss_fn = th.nn.CrossEntropyLoss()
        self.edge_loss_fn = th.nn.BCELoss()

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
        edges_per_iteration = 1
        total = 0
        start_time = time.time()

        for _ in range(epochs):
            total = 0
            print("TRAINING EPOCH", _, "FOR MODEL", self.model_params['model'], flush=True)
            for graph_file in train_dataset:
                with open(graph_file, 'rb') as graph_dict_file:
                    total += 1
                    graph_dict = pickle.load(graph_dict_file)
                    graph_dict = self.prepare_graph(graph_dict)
                    empty_graph = graph_dict['empty'].to('cuda:' + self.gpu_id)
                    full_graph = graph_dict['full'].to('cuda:' + self.gpu_id)
                    edge_labels = graph_dict['full'].edata['labels'].to('cuda:' + self.gpu_id)
                    max_iterations = graph_dict['expected'].num_edges()
                    added = True
                    adj_edge_labels = th.zeros((edge_labels.shape[0], 1)).to('cuda:' + self.gpu_id)

                    for i in range(len(edge_labels)):
                        if sum(edge_labels[i]) > 0.0:
                            adj_edge_labels[i] = 1.0

                    if full_graph.number_of_edges() > 1:
                        iterations = 0
                        current_graph = copy.deepcopy(empty_graph).to('cuda:' + self.gpu_id)

                        while ((iterations < max_iterations) and added and (full_graph.num_edges() > 1)):
                            added = False
                            iterations += 1
                            current_graph_embedding = None

                            if current_graph.num_edges():
                                current_graph_embedding = self.graph_model(current_graph, self.gpu_id)
                            else:
                                init_vector = th.zeros(self.graph_embedding_size).uniform_(-1, 1)
                                norm = init_vector.norm(p=2, dim=0, keepdim=True)
                                current_graph_embedding = init_vector.div(norm).to('cuda:' + self.gpu_id)

                            for e in range(edges_per_iteration):
                                edge_scores, add_edge_scores = self.edge_model(full_graph, current_graph_embedding)

                                if self.model_params['num_edge_types'] > 1:
                                    edge_loss = self.multi_edge_loss_fn(th.clamp(edge_scores, 0.0, 1.0), edge_labels)
                                    add_edge_loss = self.edge_loss_fn(th.clamp(add_edge_scores, 0.0, 1.0).view(-1, 1), adj_edge_labels.view(-1, 1))
                                    total_loss = edge_loss + add_edge_loss
                                else:
                                    total_loss = self.edge_loss_fn(th.clamp(edge_scores, 0.0, 1.0).view(-1, 1), edge_labels.view(-1, 1))

                                total_loss.backward()
                                th.nn.utils.clip_grad_norm(self.edge_model.parameters(), 1.0)
                                th.nn.utils.clip_grad_norm(self.graph_model.parameters(), 1.0)
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                edge_index = 0
                                feature_index = 0
                                edges = full_graph.edges()
                                edges_u = edges[0].to(th.int64)
                                edges_v = edges[1].to(th.int64)
                                max_score = -1

                                for i in range(len(edge_scores)):
                                    add_score = add_edge_scores[i]

                                    if self.model_params['num_edge_types'] > 1:
                                        if add_score > max_score:
                                            max_score = add_score
                                            edge_index = i
                                            edge = edge_scores[i]
                                            new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                            max_edge_score = -1

                                            for j in range(len(edge)):
                                                edge_type = edge[j]

                                                if edge_type > max_edge_score:
                                                    feature_index = j
                                                    max_edge_score = edge_type
                                    else:
                                        edge = edge_scores[i]

                                        if edge > max_score:
                                            max_score = edge
                                            edge_index = i
                                            new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                            max_edge_score = edge
                                            feature_index = 0

                                if ((self.model_params['num_edge_types'] > 1) and (max_score >= 0.5)) or ((self.model_params['num_edge_types'] == 1) and (max_edge_score >= 0.5)):
                                    added = True
                                    new_edge_feature[0][feature_index] = 1.
                                    edge_u = edges_u[edge_index]
                                    edge_v = edges_v[edge_index]
                                                    
                                    # add new edge
                                    current_graph.add_edges(th.tensor([edge_u]).to('cuda:' + self.gpu_id), th.tensor([edge_v]).to('cuda:' + self.gpu_id), { 'z': new_edge_feature })
                                    full_graph.remove_edges(th.tensor([edge_index]).to(th.int64).to('cuda:' + self.gpu_id))
                                    edge_labels = th.cat((edge_labels[:edge_index], edge_labels[edge_index + 1:]))
                                    adj_edge_labels = th.cat((adj_edge_labels[:edge_index], adj_edge_labels[edge_index + 1:]))

                        # clear graphs from GPU memory after use
                        del current_graph
                        del empty_graph
                        del full_graph
                        del edge_labels
                        th.cuda.empty_cache()
                        print("FINISHED GRAPH", total, "OUT OF", len(train_dataset), "IN EPOCH", _, "FOR MODEL", self.model_params['model'], flush=True)
            
            th.save(self.graph_model.state_dict(), self.model_graph_file)
            th.save(self.edge_model.state_dict(), self.model_edge_file)
        print("Finished training in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])

    def test_model(self, test_dataset):
        self.graph_model.eval()
        self.edge_model.eval()
        graph_results = []
        total_edges = {}
        edges_per_iteration = 1
        max_iterations = self.model_params['model_params']['max_iterations']
        total = 0
        start_time = time.time()
        
        for graph_file in test_dataset:
            with open(graph_file, 'rb') as graph_dict_file:
                total += 1
                graph_dict = pickle.load(graph_dict_file)
                graph_dict = self.prepare_graph(graph_dict)
                empty_graph = graph_dict['empty'].to('cuda:' + self.gpu_id)
                full_graph = graph_dict['full'].to('cuda:' + self.gpu_id)
                edge_labels = graph_dict['full'].edata['labels'].to('cuda:' + self.gpu_id)
                max_iterations = graph_dict['expected'].num_edges()
                single_total = 0
                added = True
                add_edge_labels = th.zeros((edge_labels.shape[0], 1)).to('cuda:' + self.gpu_id)

                for i in range(len(edge_labels)):
                    if sum(edge_labels[i]) > 0.0:
                        add_edge_labels[i] = 1.0

                for edge in edge_labels:
                    for e in range(len(edge)):
                        if edge[e] > 0:
                            if e in total_edges.keys():
                                total_edges[e] += 1
                            else:
                                total_edges[e] = 1
                            single_total += 1

                if full_graph.number_of_edges() > 1:
                    generated_edges = []
                    generated_edge_labels = []
                    generated_add_edges = []
                    generated_add_edge_labels = []
                    iterations = 0
                    current_graph = copy.deepcopy(empty_graph).to('cuda:' + self.gpu_id)

                    while ((iterations < max_iterations) and added):
                        added = False
                        iterations += 1
                        current_graph_embedding = None

                        if current_graph.num_edges():
                            current_graph_embedding = self.graph_model(current_graph, self.gpu_id)
                        else:
                            init_vector = th.zeros(self.graph_embedding_size).uniform_(-1, 1)
                            norm = init_vector.norm(p=2, dim=0, keepdim=True)
                            current_graph_embedding = init_vector.div(norm).to('cuda:' + self.gpu_id)

                        for e in range(edges_per_iteration):
                            edge_scores, add_edge_scores = self.edge_model(full_graph, current_graph_embedding)
                            max_score = -1
                            edge_index = 0
                            feature_index = 0
                            new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                            edges = full_graph.edges()
                            edges_u = edges[0].to(th.int64)
                            edges_v = edges[1].to(th.int64)

                            for i in range(len(edge_scores)):
                                add_score = add_edge_scores[i]

                                if self.model_params['num_edge_types'] > 1:
                                    if add_score > max_score:
                                        max_score = add_score
                                        edge_index = i
                                        edge = edge_scores[i]
                                        new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                        max_edge_score = -1

                                        for j in range(len(edge)):
                                            edge_type = edge[j]

                                            if edge_type > max_edge_score:
                                                feature_index = j
                                                max_edge_score = edge_type
                                else:
                                    edge = edge_scores[i]

                                    if edge > max_score:
                                        max_score = edge
                                        edge_index = i
                                        new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                        max_edge_score = edge
                                        feature_index = 0

                            if ((self.model_params['num_edge_types'] > 1) and (max_score >= 0.5)) or ((self.model_params['num_edge_types'] == 1) and (max_edge_score >= 0.5)):
                                added = True
                                new_edge_feature[0][feature_index] = 1.
                                edge_u = edges_u[edge_index]
                                edge_v = edges_v[edge_index]
                                            
                                # add new edge
                                current_graph.add_edges(th.tensor([edge_u]).to('cuda:' + self.gpu_id), th.tensor([edge_v]).to('cuda:' + self.gpu_id), { 'z': new_edge_feature })

                                if (self.model_params['num_edge_types'] > 1):
                                    generated_edges.append(edge)
                                    generated_edge_labels.append(edge_labels[edge_index])
                                    generated_add_edges.append(add_score)
                                    generated_add_edge_labels.append(add_edge_labels[edge_index])
                                else:
                                    generated_edges.append(edge)
                                    generated_edge_labels.append(edge_labels[edge_index])
                                    generated_add_edges.append(edge)
                                    generated_add_edge_labels.append(edge_labels[edge_index])

                                full_graph.remove_edges(th.tensor([edge_index]).to(th.int64).to('cuda:' + self.gpu_id))
                                edge_labels = th.cat((edge_labels[:edge_index], edge_labels[edge_index + 1:]))
                                add_edge_labels = th.cat((add_edge_labels[:edge_index], add_edge_labels[edge_index + 1:]))
                                #one_edge.append(edge[0].item())
                            else:
                                pass
                                #no_edge.append(edge[1].item())

                    if len(generated_edges) > 0:
                        print("Added edges", len(generated_edges))
                        generated_edges = th.stack(generated_edges)
                        generated_edge_labels = th.stack(generated_edge_labels)
                        generated_add_edges = th.stack(generated_add_edges)
                        generated_add_edge_labels = th.stack(generated_add_edge_labels)
                        graph_result = [] # First values prediction, second value label

                        for i in range(len(generated_edges)):
                            if self.model_params['num_edge_types'] > 1:
                                gen_edge = list(generated_edges[i])
                                gen_edge_label = list(generated_edge_labels[i])
                                add_edge = generated_add_edges[i]
                                add_edge_label = generated_add_edge_labels[i]
                                max_edge = max(gen_edge)
                                max_label = max(gen_edge_label)

                                if add_edge >= 0.5:
                                    if add_edge_label >= 0.5:
                                        graph_result.append([gen_edge.index(max_edge), gen_edge_label.index(max_label), single_total])
                                    else:
                                        graph_result.append([gen_edge.index(max_edge), -1, single_total])
                                else:
                                    if add_edge_label >= 0.5:
                                        graph_result.append([-1, gen_edge_label.index(max_label), single_total])
                                    else:
                                        graph_result.append([-1, -1, single_total])
                            else:
                                gen_edge = generated_edges[i]
                                gen_edge_label = generated_edge_labels[i]

                                if gen_edge >= 0.5:
                                    if gen_edge_label >= 0.5:
                                        graph_result.append([0, 0, single_total])
                                    else:
                                        graph_result.append([0, -1, single_total])
                                else:
                                    if gen_edge_label >= 0.5:
                                        graph_result.append([-1, 0, single_total])
                                    else:
                                        graph_result.append([-1, -1, single_total])

                        graph_results.append(graph_result)
                    else:
                        #print(self.model_params['model'], "NO EDGES ADDED", graph_file)
                        graph_results.append([])

                    # clear graphs from GPU memory after use
                    del empty_graph
                    del current_graph
                    del full_graph
                    del edge_labels
                    del generated_edges
                    del generated_edge_labels
                    th.cuda.empty_cache()
                    print("FINISHED GRAPH", total, "OUT OF", len(test_dataset), "FOR MODEL", self.model_params['model'], flush=True)
        print("Finished testing in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"], len(test_dataset))

        self.results.add_metrics(self.model_params['model'], graph_results, total_edges)