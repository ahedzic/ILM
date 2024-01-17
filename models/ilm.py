from . import graph_model
from . import edge_score_ilm, HGAT_meta
import copy
import dgl
import itertools
import os.path
import pickle
import torch as th
import random
import time

class ILM(graph_model.BaseGraphModel):
    def __init__(self, model_params, dataset, results, gpu_id):
        super(ILM, self).__init__(model_params, dataset, results, gpu_id)

    def initialize_model(self):
        node_feature_size = self.model_params['node_feature_size']
        num_edge_types = self.model_params['num_edge_types']
        attention_heads = self.model_params['model_params']['attention_heads']
        self.graph_embedding_size = self.model_params['model_params']['graph_embedding_size']
        self.graph_model = HGAT_meta.HGAT_Meta(node_feature_size, attention_heads, num_edge_types, self.graph_embedding_size).to('cuda:' + self.gpu_id)
        self.edge_model = edge_score_ilm.EdgeScore((2 * node_feature_size) + self.graph_embedding_size, num_edge_types, self.model_params['model_params']['graph_embedding_size'], node_feature_size, self.model_params['model_params']['group_count'], self.gpu_id).to('cuda:' + self.gpu_id)
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
        max_iterations = self.model_params['model_params']['max_iterations']
        edges_per_iteration = 1
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
                    adj_edge_labels = th.zeros((edge_labels.shape[0], 1)).to('cuda:' + self.gpu_id)

                    for i in range(len(edge_labels)):
                        if sum(edge_labels[i]) > 0.0:
                            adj_edge_labels[i] = 1.0

                    if full_graph.number_of_edges() > 1:
                        iterations = 0
                        current_graph = copy.deepcopy(empty_graph).to('cuda:' + self.gpu_id)

                        if True:
                            while (iterations < max_iterations):
                                iterations += 1
                                current_graph_embedding = None
                                if th.isnan(current_graph.ndata['feat']).any():
                                    print("Current graph invalid")

                                if current_graph.num_edges():
                                    current_graph.ndata['feat'] = th.clamp(current_graph.ndata['feat'], -1.0, 1.0)
                                    current_graph_embedding, node_embeddings = self.graph_model(current_graph, self.gpu_id)
                                else:
                                    init_vector = th.zeros(self.graph_embedding_size).uniform_(-1, 1)
                                    norm = init_vector.norm(p=2, dim=0, keepdim=True)
                                    if norm > 0.0:
                                        current_graph_embedding = init_vector.div(norm).to('cuda:' + self.gpu_id)
                                    else:
                                        current_graph_embedding = init_vector
                                    node_embeddings = th.zeros(full_graph.number_of_nodes(), self.graph_embedding_size).to('cuda:' + self.gpu_id)

                                    for i in range(len(node_embeddings)):
                                        node_embeddings[i] = current_graph_embedding

                                current_graph = copy.deepcopy(empty_graph).to('cuda:' + self.gpu_id)

                                for e in range(edges_per_iteration):
                                    if th.isnan(full_graph.ndata['feat']).any():
                                        print("Current graph invalid")
                                    edge_scores, meta_scores, add_edge_scores = self.edge_model(full_graph, current_graph_embedding, node_embeddings)

                                    if self.model_params['num_edge_types'] > 1:
                                        edge_loss = self.multi_edge_loss_fn(th.clamp(edge_scores, 0.0, 1.0), th.clamp(edge_labels, 0.0, 1.0))
                                        add_edge_loss = self.edge_loss_fn(th.clamp(add_edge_scores, 0.0, 1.0).view(-1, 1), th.clamp(adj_edge_labels, 0.0, 1.0).view(-1, 1))
                                        total_loss = edge_loss + add_edge_loss# + g1_loss + g2_loss
                                    else:
                                        total_loss = self.edge_loss_fn(th.clamp(edge_scores, 0.0, 1.0).view(-1, 1), th.clamp(edge_labels, 0.0, 1.0).view(-1, 1))# + g1_loss + g2_loss

                                    total_loss.backward()
                                    th.nn.utils.clip_grad_norm_(self.edge_model.parameters(), 1.0)
                                    th.nn.utils.clip_grad_norm_(self.graph_model.parameters(), 1.0)
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                                    edge_index = 0
                                    feature_index = 0
                                    edges = full_graph.edges()
                                    edges_u = edges[0].to(th.int64)
                                    edges_v = edges[1].to(th.int64)

                                    for i in range(len(edge_scores)):
                                        edge = edge_scores[i]
                                        add_score = add_edge_scores[i]
                                        new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                        edge_index = i
                                        max_score = -1

                                        for j in range(len(edge)):
                                            edge_type = edge[j]

                                            if edge_type > max_score:
                                                max_score = edge_type
                                                feature_index = j

                                        if ((self.model_params['num_edge_types'] > 1) and (add_score >= 0.5)) or ((self.model_params['num_edge_types'] == 1) and (edge >= 0.5)):
                                            new_edge_feature[0][feature_index] = 1.
                                            edge_u = edges_u[edge_index]
                                            edge_v = edges_v[edge_index]
                                                    
                                            # add new edge
                                            current_graph.add_edges(th.tensor([edge_u]).to('cuda:' + self.gpu_id), th.tensor([edge_v]).to('cuda:' + self.gpu_id), { 'z': new_edge_feature })
                                
                                del self.edge_model.graph_embedding_matrix
                                del edge_scores, meta_scores, add_edge_scores

                                if current_graph.num_edges():
                                    del current_graph_embedding, node_embeddings
                                th.cuda.empty_cache()

                        # clear graphs from GPU memory after use
                        del current_graph
                        del empty_graph
                        del full_graph
                        del edge_labels
                        del adj_edge_labels
                        th.cuda.empty_cache()
                    print("FINISHED GRAPH", total, "OUT OF", len(train_dataset), "IN EPOCH", _, "FOR MODEL", self.model_params['model'], flush=True)
            
            th.save(self.graph_model.state_dict(), self.model_graph_file)
            th.save(self.edge_model.state_dict(), self.model_edge_file)
        print("Finished training in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])

    def test_model(self, test_dataset):
        graph_results = []
        total_edges = {0: 0}
        edges_per_iteration = 1
        max_iterations = self.model_params['model_params']['max_iterations']
        total_edge_distributions = []
        meta_distributions = []
        meta_w_distributions = []
        total_mismatch_12_add = 0.0
        total_mismatch_23_add = 0.0
        total_mismatch_12_remove = 0.0
        total_mismatch_23_remove = 0.0
        total_graphs = 0
        start_time = time.time()
        
        for graph_file in test_dataset:
            with open(graph_file, 'rb') as graph_dict_file:
                graph_dict = pickle.load(graph_dict_file)
                graph_dict = self.prepare_graph(graph_dict)
                empty_graph = graph_dict['empty'].to('cuda:' + self.gpu_id)
                full_graph = graph_dict['full'].to('cuda:' + self.gpu_id)
                edge_labels = graph_dict['full'].edata['labels']
                single_total = 0
                add_edge_labels = th.zeros((edge_labels.shape[0], 1)).to('cuda:' + self.gpu_id)
                graph_result = [] # First values prediction, second value label
                edge_distributions = []

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
                        
                '''if empty_graph.number_of_edges() > 0:
                    single_total += empty_graph.number_of_edges()

                    for i in range(empty_graph.number_of_edges()):
                        graph_result.append([0, 0, single_total])
                        total_edges[0] += 1'''

                if full_graph.number_of_edges() > 1:
                    generated_edges = []
                    generated_edge_labels = []
                    generated_add_edges = []
                    generated_add_edge_labels = []
                    iterations = 0
                    current_graph = copy.deepcopy(empty_graph).to('cuda:' + self.gpu_id)
                    no_edge = []
                    one_edge = []
                    edge_matches = []
                    total_graphs += 1
                    print("TESTING", total_graphs, "OUT OF", len(test_dataset), "FOR MODEL", self.model_params['model'])
                    
                    while (iterations < max_iterations):
                        edge_match = []
                        iterations += 1
                        current_graph_embedding = None

                        if current_graph.num_edges():
                            current_graph_embedding, node_embeddings = self.graph_model(current_graph, self.gpu_id)
                        else:
                            init_vector = th.zeros(self.graph_embedding_size).uniform_(-1, 1)
                            norm = init_vector.norm(p=2, dim=0, keepdim=True)
                            current_graph_embedding = init_vector.div(norm).to('cuda:' + self.gpu_id)
                            node_embeddings = th.zeros(full_graph.number_of_nodes(), self.graph_embedding_size).to('cuda:' + self.gpu_id)

                            for i in range(len(node_embeddings)):
                                node_embeddings[i] = current_graph_embedding

                        generated_edges = []
                        generated_edge_labels = []
                        generated_add_edges = []
                        generated_add_edge_labels = []
                        current_graph = copy.deepcopy(empty_graph).to('cuda:' + self.gpu_id)

                        for e in range(edges_per_iteration):
                            edge_scores, meta_scores, add_edge_scores = self.edge_model(full_graph, current_graph_embedding, node_embeddings)
                            edge_index = 0
                            feature_index = 0
                            edges = full_graph.edges()
                            edges_u = edges[0].to(th.int64)
                            edges_v = edges[1].to(th.int64)

                            for i in range(len(edge_scores)):
                                edge = edge_scores[i]
                                add_score = add_edge_scores[i]
                                new_edge_feature = th.tensor([[0.] * self.model_params['num_edge_types']]).to('cuda:' + self.gpu_id)
                                edge_index = i
                                max_score = -1

                                if self.model_params['num_edge_types'] > 1:
                                    for j in range(len(edge)):
                                        edge_type = edge[j]

                                        if edge_type > max_score:
                                            max_score = edge_type
                                            feature_index = j
                                else:
                                    edge_type = edge

                                if ((self.model_params['num_edge_types'] > 1) and (add_score >= 0.5)) or ((self.model_params['num_edge_types'] == 1) and (edge >= 0.5)):
                                    new_edge_feature[0][feature_index] = 1.
                                    edge_u = edges_u[edge_index]
                                    edge_v = edges_v[edge_index]
                                    edge_match.append(1)
                                            
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
                                    #one_edge.append(edge[0].item())
                                else:
                                    edge_match.append(0)
                                    #no_edge.append(edge[1].item())

                            edge_matches.append(edge_match)
                            del self.edge_model.graph_embedding_matrix
                            del self.edge_model.weighted_embedding
                            del edge_scores, add_edge_scores
                            th.cuda.empty_cache()

                        '''if iterations == 3:
                            print("One edge")
                            for val in one_edge:
                                print(val)
                            print("No edge")
                            for val in no_edge:
                                print(val)'''


                        edge_distributions.append(current_graph.number_of_edges())
                        total = float(len(meta_scores))
                        glo = 0.0
                        loc = 0.0
                        diff = 0.0
                        sim = 0.0
                        #group = 0.0
                        glo_w = 0.0
                        loc_w = 0.0
                        diff_w = 0.0
                        sim_w = 0.0
                        #group_w = 0.0

                        for meta in meta_scores:
                            meta = list(meta)
                            max_index = meta.index(max(meta))
                            glo_w += meta[0]
                            loc_w += meta[1]
                            diff_w += meta[2]
                            #sim_w += meta[3]
                            #group_w += meta[3]

                            if max_index == 0:
                                glo += 1.0
                            elif max_index == 1:
                                loc += 1.0
                            elif max_index == 2:
                                diff += 1.0
                            #elif max_index == 3:
                            #    sim += 1.0
                            #elif max_index == 3:
                            #    group += 1.0

                        del meta_scores

                        if current_graph.num_edges():
                            del current_graph_embedding, node_embeddings

                        th.cuda.empty_cache()

                    mismatch_12_add = 0.0
                    mismatch_23_add = 0.0
                    mismatch_12_remove = 0.0
                    mismatch_23_remove = 0.0

                    for i in range(len(edge_matches[0])):
                        if edge_matches[0][i] == 0 and edge_matches[1][i] == 1:
                            mismatch_12_add += 1.0
                        if edge_matches[0][i] == 1 and edge_matches[1][i] == 0:
                            mismatch_12_remove += 1.0
                        if edge_matches[1][i] == 0 and edge_matches[2][i] == 1:
                            mismatch_23_add += 1.0
                        if edge_matches[1][i] == 1 and edge_matches[2][i] == 0:
                            mismatch_23_remove += 1.0

                    total_mismatch_12_add += mismatch_12_add / float(len(edge_matches[0]))
                    total_mismatch_23_add += mismatch_23_add / float(len(edge_matches[0]))
                    total_mismatch_12_remove += mismatch_12_remove / float(len(edge_matches[0]))
                    total_mismatch_23_remove += mismatch_23_remove / float(len(edge_matches[0]))

                    meta_distributions.append([glo / total, loc / total, diff / total])
                    meta_w_distributions.append([glo_w / total, loc_w / total, diff_w / total])

                    if len(generated_edges) > 0:
                        generated_edges = th.stack(generated_edges)
                        generated_edge_labels = th.stack(generated_edge_labels)
                        generated_add_edges = th.stack(generated_add_edges)
                        generated_add_edge_labels = th.stack(generated_add_edge_labels)

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
                                        graph_result.append([0, 0, single_total])
                                        #graph_result.append([gen_edge.index(max_edge), gen_edge_label.index(max_label), single_total])
                                    else:
                                        graph_result.append([0, -1, single_total])
                                        #graph_result.append([gen_edge.index(max_edge), -1, single_total])
                                else:
                                    if add_edge_label >= 0.5:
                                        graph_result.append([-1, 0, single_total])
                                        #graph_result.append([-1, gen_edge_label.index(max_label), single_total])
                                    else:
                                        graph_result.append([-1, -1, single_total])
                                        #graph_result.append([-1, -1, single_total])
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
                    del add_edge_labels
                    del generated_edges
                    del generated_edge_labels
                    th.cuda.empty_cache()

                if len(edge_distributions):
                    total_edge_distributions.append(edge_distributions)

        print("Finished testing in time ------- %s ------- seconds" % (time.time() - start_time), "FOR MODEL", self.model_params['model'], self.model_params["dataset_name"])
        final_edge_dist = [0.0] * max_iterations
        final_meta_dist = [0.0] * len(meta_distributions[0])
        final_meta_w_dist = [0.0] * len(meta_distributions[0])

        for edge_dist in total_edge_distributions:
            for i in range(max_iterations):
                final_edge_dist[i] += edge_dist[i]

        for i in range(max_iterations):
            final_edge_dist[i] /= len(total_edge_distributions)

        for meta_dist in meta_distributions:
            for i in range(len(final_meta_dist)):
                final_meta_dist[i] += meta_dist[i]

        for meta_w_dist in meta_w_distributions:
            for i in range(len(final_meta_w_dist)):      
                final_meta_w_dist[i] += meta_w_dist[i] 

        for i in range(len(final_meta_dist)):
            final_meta_dist[i] /= len(meta_distributions[0])

        for i in range(len(final_meta_w_dist)):
            final_meta_w_dist[i] /= len(meta_w_distributions[0])

        print("Iteration 1 to 2 percent change add", total_mismatch_12_add / float(len(test_dataset)), self.model_params["dataset_name"])
        print("Iteration 2 to 3 percent change add", total_mismatch_23_add / float(len(test_dataset)), self.model_params["dataset_name"])
        print("Iteration 1 to 2 percent change remove", total_mismatch_12_remove / float(len(test_dataset)), self.model_params["dataset_name"])
        print("Iteration 2 to 3 percent change remove", total_mismatch_23_remove / float(len(test_dataset)), self.model_params["dataset_name"])
        print("Edge distributions", final_edge_dist, self.model_params["dataset_name"])
        #print("Meta distributions", final_meta_dist, self.model_params["dataset_name"])
        final_meta_w_dist = th.tensor(final_meta_w_dist)
        #print("Meta weight distributions", final_meta_w_dist, self.model_params["dataset_name"])
        #meta_norm = final_meta_w_dist.norm(p=2, dim=0, keepdim=True)
        meta_normalized = final_meta_w_dist.div(sum(final_meta_w_dist))
        print("Meta weight distributions normalized", meta_normalized, self.model_params["dataset_name"])
        self.results.add_metrics(self.model_params['model'], graph_results, total_edges)