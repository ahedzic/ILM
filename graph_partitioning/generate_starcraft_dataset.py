from generate_dataset import generate_dataset
import os
import pickle
import random
import glob
import dgl
import torch

def dataset_load():
    graph_list = glob.glob(os.path.join('../starcraft_graphs/*.pkl'))
    random.shuffle(graph_list)
    graphs = []

    for graph_file in graph_list:
        with open(graph_file, 'rb') as graph_dict_file:
            graph_dict = pickle.load(graph_dict_file)
            graphs.append(graph_dict['expected'])
        
    return graphs

def generate_dataset(dataset_name, graphs):
    processed_graphs = []
    processed_graphs_true = []
    processed_graphs_25_edge = []
    processed_graphs_50_edge = []
    processed_graphs_75_edge = []
    processed_graphs_90_edge = []
    processed_graphs_25_node = []
    processed_graphs_50_node = []
    processed_graphs_75_node = []
    processed_graphs_90_node = []

    edge_type_counts = {0: 0,
                        1: 0,
                        2: 0}
    unique_edge_type_counts = {0: 0,
                        1: 0,
                        2: 0}
    feature_edge_type_counts = {0: 0,
                        1: 0,
                        2: 0}

    for graph in graphs:
        graph_dict = {}
        graph_dict_true = {}
        graph_dict_25_edge = {}
        graph_dict_50_edge = {}
        graph_dict_75_edge = {}
        graph_dict_90_edge = {}
        graph_dict_25_node = {}
        graph_dict_50_node = {}
        graph_dict_75_node = {}
        graph_dict_90_node = {}
        
        graph_dict['gnn_feature'] = graph.ndata['feat']
        remove_edges = []

        for edge_index in range(len(graph.edata['z'])):
            if torch.argmax(graph.edata['z'][edge_index]).item() == 7:
                remove_edges.append(edge_index)
            if torch.argmax(graph.edata['z'][edge_index]).item() == 6:
                remove_edges.append(edge_index)
            
        graph = dgl.remove_edges(graph, torch.tensor(remove_edges))

        edge_features = torch.zeros(graph.edata['z'].shape[0], 3)

        for edge_index in range(len(graph.edata['z'])):
            argmax = int(torch.argmax(graph.edata['z'][edge_index]).item())

            if (argmax == 5):# or (argmax == 6):
                edge_features[edge_index][2] = 1.0
                edge_type_counts[2] += 1
            elif argmax == 1:
                edge_features[edge_index][1] = 1.0
                edge_type_counts[1] += 1
            elif argmax == 0:
                edge_features[edge_index][0] = 1.0
                edge_type_counts[0] += 1
            else:
                print("extraneous edge", argmax)

        pos_edges_and_types = []
        neg_edges_and_types = []
        total_pos_edges = len(graph.edges()[0])
        unique_edges = set()

        for i in range(total_pos_edges):
            sub = int(graph.edges()[0][i].item())
            obj = int(graph.edges()[1][i].item())
            if (sub, obj) not in unique_edges:
                unique_edges.add((sub, obj))
                argmax = int(torch.argmax(edge_features[i]).item())
                pos_edges_and_types.append(((sub, obj), argmax))
                unique_edge_type_counts[argmax] += 1

        #for feat in features:
        #    argmax = int(torch.argmax(torch.tensor(feat)).item())
        #    feature_edge_type_counts[argmax] += 1

        total_pos_edges = len(pos_edges_and_types)
        negative_graph_edges = dgl.sampling.global_uniform_negative_sampling(graph, total_pos_edges)
        total_neg_edges = len(negative_graph_edges[0])

        for i in range(min([total_pos_edges, total_neg_edges])):
            sub = int(negative_graph_edges[0][i].item())
            obj = int(negative_graph_edges[1][i].item())
            neg_edges_and_types.append(((sub, obj), random.randint(0, 2)))

        random.shuffle(pos_edges_and_types)
        random.shuffle(neg_edges_and_types)

        pos_edges = []
        neg_edges = []
        pos_types = []
        neg_types = []

        for edge in pos_edges_and_types:
            pos_edges.append(edge[0])
            pos_types.append(edge[1])

        for edge in neg_edges_and_types:
            neg_edges.append(edge[0])
            neg_types.append(edge[1])

        graph_dict['pos_edges'] = pos_edges
        graph_dict['neg_edges'] = neg_edges
        graph_dict['pos_types'] = pos_types
        graph_dict['neg_types'] = neg_types
        graph_dict['node_count'] = graph.number_of_nodes()
        processed_graphs.append(graph_dict)

        graph_dict_true['pos_edges'] = pos_edges
        graph_dict_true['given_edges'] = []
        graph_dict_true['neg_edges'] = neg_edges
        graph_dict_true['pos_types'] = pos_types
        graph_dict_true['neg_types'] = neg_types
        graph_dict_true['node_count'] = graph.number_of_nodes()
        graph_dict_true['gnn_feature'] = graph.ndata['feat']
        processed_graphs_true.append(graph_dict_true)

        edges_retained_25 = max([1, int(len(pos_edges) * 0.25)])
        edges_retained_50 = max([1, int(len(pos_edges) * 0.50)])
        edges_retained_75 = max([1, int(len(pos_edges) * 0.75)])
        edges_retained_90 = max([1, int(len(pos_edges) * 0.90)])
        
        graph_dict_25_edge['pos_edges'] = pos_edges[edges_retained_25:]
        graph_dict_25_edge['given_edges'] = pos_edges[:edges_retained_25]
        graph_dict_25_edge['neg_edges'] = neg_edges[edges_retained_25:]
        graph_dict_25_edge['pos_types'] = pos_types[edges_retained_25:]
        graph_dict_25_edge['given_types'] = pos_types[:edges_retained_25]
        graph_dict_25_edge['neg_types'] = neg_types[edges_retained_25:]
        graph_dict_25_edge['node_count'] = graph.number_of_nodes()
        graph_dict_25_edge['gnn_feature'] = graph.ndata['feat']
        processed_graphs_25_edge.append(graph_dict_25_edge)

        graph_dict_50_edge['pos_edges'] = pos_edges[edges_retained_50:]
        graph_dict_50_edge['given_edges'] = pos_edges[:edges_retained_50]
        graph_dict_50_edge['neg_edges'] = neg_edges[edges_retained_50:]
        graph_dict_50_edge['pos_types'] = pos_types[edges_retained_50:]
        graph_dict_50_edge['given_types'] = pos_types[:edges_retained_50]
        graph_dict_50_edge['neg_types'] = neg_types[edges_retained_50:]
        graph_dict_50_edge['node_count'] = graph.number_of_nodes()
        graph_dict_50_edge['gnn_feature'] = graph.ndata['feat']
        processed_graphs_50_edge.append(graph_dict_50_edge)

        graph_dict_75_edge['pos_edges'] = pos_edges[edges_retained_75:]
        graph_dict_75_edge['given_edges'] = pos_edges[:edges_retained_75]
        graph_dict_75_edge['neg_edges'] = neg_edges[edges_retained_75:]
        graph_dict_75_edge['pos_types'] = pos_types[edges_retained_75:]
        graph_dict_75_edge['given_types'] = pos_types[:edges_retained_75]
        graph_dict_75_edge['neg_types'] = neg_types[edges_retained_75:]
        graph_dict_75_edge['node_count'] = graph.number_of_nodes()
        graph_dict_75_edge['gnn_feature'] = graph.ndata['feat']
        processed_graphs_75_edge.append(graph_dict_75_edge)

        graph_dict_90_edge['pos_edges'] = pos_edges[edges_retained_90:]
        graph_dict_90_edge['given_edges'] = pos_edges[:edges_retained_90]
        graph_dict_90_edge['neg_edges'] = neg_edges[edges_retained_90:]
        graph_dict_90_edge['pos_types'] = pos_types[edges_retained_90:]
        graph_dict_90_edge['given_types'] = pos_types[:edges_retained_90]
        graph_dict_90_edge['neg_types'] = neg_types[edges_retained_90:]
        graph_dict_90_edge['node_count'] = graph.number_of_nodes()
        graph_dict_90_edge['gnn_feature'] = graph.ndata['feat']
        processed_graphs_90_edge.append(graph_dict_90_edge)

        num_nodes = graph.number_of_nodes()
        nodes_retained_25 = max([1, int(num_nodes * 0.25)])
        nodes_to_remove_25 = random.sample(range(num_nodes), (num_nodes - nodes_retained_25))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for edge_i in range(len(pos_edges)):
            if (pos_edges[edge_i][0] not in nodes_to_remove_25) and (pos_edges[edge_i][1] not in nodes_to_remove_25):
                given_edges.append(pos_edges[edge_i])
                given_types.append(pos_types[edge_i])
            else:
                positive_edges.append(pos_edges[edge_i])
                positive_types.append(pos_types[edge_i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_25_node['pos_edges'] = positive_edges
        graph_dict_25_node['given_edges'] = given_edges
        graph_dict_25_node['neg_edges'] = negative_edges
        graph_dict_25_node['pos_types'] = positive_types
        graph_dict_25_node['given_types'] = given_types
        graph_dict_25_node['node_count'] = graph.number_of_nodes()
        graph_dict_25_node['gnn_feature'] = graph.ndata['feat']
        processed_graphs_25_node.append(graph_dict_25_node)

        nodes_retained_50 = max([1, int(num_nodes * 0.50)])
        nodes_to_remove_50 = random.sample(range(num_nodes), (num_nodes - nodes_retained_50))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for edge_i in range(len(pos_edges)):
            if (pos_edges[edge_i][0] not in nodes_to_remove_50) and (pos_edges[edge_i][1] not in nodes_to_remove_50):
                given_edges.append(pos_edges[edge_i])
                given_types.append(pos_types[edge_i])
            else:
                positive_edges.append(pos_edges[edge_i])
                positive_types.append(pos_types[edge_i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_50_node['pos_edges'] = positive_edges
        graph_dict_50_node['given_edges'] = given_edges
        graph_dict_50_node['neg_edges'] = negative_edges
        graph_dict_50_node['pos_types'] = positive_types
        graph_dict_50_node['given_types'] = given_types
        graph_dict_50_node['node_count'] = graph.number_of_nodes()
        graph_dict_50_node['gnn_feature'] = graph.ndata['feat']
        processed_graphs_50_node.append(graph_dict_50_node)

        nodes_retained_75 = max([1, int(num_nodes * 0.75)])
        nodes_to_remove_75 = random.sample(range(num_nodes), (num_nodes - nodes_retained_75))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for edge_i in range(len(pos_edges)):
            if (pos_edges[edge_i][0] not in nodes_to_remove_75) and (pos_edges[edge_i][1] not in nodes_to_remove_75):
                given_edges.append(pos_edges[edge_i])
                given_types.append(pos_types[edge_i])
            else:
                positive_edges.append(pos_edges[edge_i])
                positive_types.append(pos_types[edge_i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_75_node['pos_edges'] = positive_edges
        graph_dict_75_node['given_edges'] = given_edges
        graph_dict_75_node['neg_edges'] = negative_edges
        graph_dict_75_node['pos_types'] = positive_types
        graph_dict_75_node['given_types'] = given_types
        graph_dict_75_node['node_count'] = graph.number_of_nodes()
        graph_dict_75_node['gnn_feature'] = graph.ndata['feat']
        processed_graphs_75_node.append(graph_dict_75_node)

        nodes_retained_90 = max([1, int(num_nodes * 0.90)])
        nodes_to_remove_90 = random.sample(range(num_nodes), (num_nodes - nodes_retained_90))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for edge_i in range(len(pos_edges)):
            if (pos_edges[edge_i][0] not in nodes_to_remove_90) and (pos_edges[edge_i][1] not in nodes_to_remove_90):
                given_edges.append(pos_edges[edge_i])
                given_types.append(pos_types[edge_i])
            else:
                positive_edges.append(pos_edges[edge_i])
                positive_types.append(pos_types[edge_i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_90_node['pos_edges'] = positive_edges
        graph_dict_90_node['given_edges'] = given_edges
        graph_dict_90_node['neg_edges'] = negative_edges
        graph_dict_90_node['pos_types'] = positive_types
        graph_dict_90_node['given_types'] = given_types
        graph_dict_90_node['node_count'] = graph.number_of_nodes()
        graph_dict_90_node['gnn_feature'] = graph.ndata['feat']
        processed_graphs_90_node.append(graph_dict_90_node)

    print("Total graphs", len(graphs))
    print("Edge counts", edge_type_counts)
    print("Unique edge count", unique_edge_type_counts)
    #print("Features edge counts", feature_edge_type_counts)

    # Partitions were already shuffled just take proportionate slice out of array
    # Use 85/5/10% Train/Validation/Test Split
    train_size = int(0.85 * len(graphs))
    validation_size = int(0.05 * len(graphs))
    test_size = int(0.1 * len(graphs))
    simple_train_graphs = processed_graphs[:train_size]
    simple_validation_graphs = processed_graphs[train_size:train_size+validation_size]
    simple_test_graphs = processed_graphs[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs = {
        'train': simple_train_graphs,
        'valid': simple_validation_graphs,
        'test': simple_test_graphs
    }

    print("Total graphs after", len(simple_train_graphs)+len(simple_validation_graphs)+len(simple_test_graphs))

    simple_file = open(dataset_name+'_graphs.pkl', 'wb')
    pickle.dump(simple_graphs, simple_file)
    simple_file.close()

    true_train_graphs = processed_graphs_true[:train_size]
    true_validation_graphs = processed_graphs_true[train_size:train_size+validation_size]
    true_test_graphs = processed_graphs_true[train_size+validation_size:train_size+validation_size+test_size]

    true_graphs = {
        'train': true_train_graphs,
        'valid': true_validation_graphs,
        'test': true_test_graphs
    }
    
    true_file = open(dataset_name+'_true_graphs.pkl', 'wb')
    pickle.dump(true_graphs, true_file)
    true_file.close()

    train_graphs_25_edge = processed_graphs_25_edge[:train_size]
    validation_graphs_25_edge = processed_graphs_25_edge[train_size:train_size+validation_size]
    test_graphs_25_edge = processed_graphs_25_edge[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_25_edge = {
        'train': train_graphs_25_edge,
        'valid': validation_graphs_25_edge,
        'test': test_graphs_25_edge
    }
    
    simple_file_25_edge = open(dataset_name+'_25_edge_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_25_edge, simple_file_25_edge)
    simple_file_25_edge.close()

    train_graphs_50_edge = processed_graphs_50_edge[:train_size]
    validation_graphs_50_edge = processed_graphs_50_edge[train_size:train_size+validation_size]
    test_graphs_50_edge = processed_graphs_50_edge[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_50_edge = {
        'train': train_graphs_50_edge,
        'valid': validation_graphs_50_edge,
        'test': test_graphs_50_edge
    }
    
    simple_file_50_edge = open(dataset_name+'_50_edge_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_50_edge, simple_file_50_edge)
    simple_file_50_edge.close()

    train_graphs_75_edge = processed_graphs_75_edge[:train_size]
    validation_graphs_75_edge = processed_graphs_75_edge[train_size:train_size+validation_size]
    test_graphs_75_edge = processed_graphs_75_edge[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_75_edge = {
        'train': train_graphs_75_edge,
        'valid': validation_graphs_75_edge,
        'test': test_graphs_75_edge
    }
    
    simple_file_75_edge = open(dataset_name+'_75_edge_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_75_edge, simple_file_75_edge)
    simple_file_75_edge.close()

    train_graphs_90_edge = processed_graphs_90_edge[:train_size]
    validation_graphs_90_edge = processed_graphs_90_edge[train_size:train_size+validation_size]
    test_graphs_90_edge = processed_graphs_90_edge[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_90_edge = {
        'train': train_graphs_90_edge,
        'valid': validation_graphs_90_edge,
        'test': test_graphs_90_edge
    }
    
    simple_file_90_edge = open(dataset_name+'_90_edge_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_90_edge, simple_file_90_edge)
    simple_file_90_edge.close()

    train_graphs_25_node = processed_graphs_25_node[:train_size]
    validation_graphs_25_node = processed_graphs_25_node[train_size:train_size+validation_size]
    test_graphs_25_node = processed_graphs_25_node[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_25_node = {
        'train': train_graphs_25_node,
        'valid': validation_graphs_25_node,
        'test': test_graphs_25_node
    }
    
    simple_file_25_node = open(dataset_name+'_25_node_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_25_node, simple_file_25_node)
    simple_file_25_node.close()

    train_graphs_50_node = processed_graphs_50_node[:train_size]
    validation_graphs_50_node = processed_graphs_50_node[train_size:train_size+validation_size]
    test_graphs_50_node = processed_graphs_50_node[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_50_node = {
        'train': train_graphs_50_node,
        'valid': validation_graphs_50_node,
        'test': test_graphs_50_node
    }
    
    simple_file_50_node = open(dataset_name+'_50_node_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_50_node, simple_file_50_node)
    simple_file_50_node.close()

    train_graphs_75_node = processed_graphs_75_node[:train_size]
    validation_graphs_75_node = processed_graphs_75_node[train_size:train_size+validation_size]
    test_graphs_75_node = processed_graphs_75_node[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_75_node = {
        'train': train_graphs_75_node,
        'valid': validation_graphs_75_node,
        'test': test_graphs_75_node
    }
    
    simple_file_75_node = open(dataset_name+'_75_node_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_75_node, simple_file_75_node)
    simple_file_75_node.close()

    train_graphs_90_node = processed_graphs_90_node[:train_size]
    validation_graphs_90_node = processed_graphs_90_node[train_size:train_size+validation_size]
    test_graphs_90_node = processed_graphs_90_node[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs_90_node = {
        'train': train_graphs_90_node,
        'valid': validation_graphs_90_node,
        'test': test_graphs_90_node
    }
    
    simple_file_90_node = open(dataset_name+'_90_node_graphs.pkl', 'wb')
    pickle.dump(simple_graphs_90_node, simple_file_90_node)
    simple_file_90_node.close()

def main():
    graphs = dataset_load()
    generate_dataset('starcraft', graphs)

if __name__ == '__main__':
    main()