import copy
import dgl
import random

class BaseGraphModel():
    def __init__(self, model_params, dataset, results, gpu_id):
        self.model_params = model_params
        self.dataset = dataset
        self.results = results
        self.gpu_id = gpu_id
        self.edge_percentage = self.model_params["edge_percentage"]

    def initialize_model(self):
        pass

    def prepare_graph(self, graph_dict):
        if self.edge_percentage > 0.0:
            new_graph_dict = {}
            current_graph = copy.deepcopy(graph_dict['expected'])
            full_graph = copy.deepcopy(graph_dict['full'])
            left_edges = int(self.edge_percentage * (current_graph.number_of_edges()) + 1)

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

        return graph_dict

    def train_model(self, train_dataset):
        pass

    def test_model(self, test_dataset):
        pass