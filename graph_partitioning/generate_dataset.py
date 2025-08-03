import dgl
import torch as th
import os
import pickle
import random

from os.path import exists

def generate_full_dataset(dataset_name, whole_graph):
    total_edges = whole_graph.number_of_edges()
    print("Original Graph Edges", total_edges)

    feature_embeddings = {}
    feature_embeddings['entity_embedding'] = whole_graph.ndata['feat']

    features_file = open('gnn_feature', 'wb')
    th.save(feature_embeddings, features_file)
    features_file.close()
    
    pos_edges = []
    neg_edges = []
    total_pos_edges = len(whole_graph.edges()[0])
    negative_graph_edges = dgl.sampling.global_uniform_negative_sampling(whole_graph, total_pos_edges)
    total_neg_edges = len(negative_graph_edges[0])
    subjects = whole_graph.edges()[0]
    objects = whole_graph.edges()[1]
    neg_subjects = negative_graph_edges[0]
    neg_objects = negative_graph_edges[1]

    for i in range(min([total_pos_edges, total_neg_edges])):
        if i % 10000 == 0:
            print("Finished", i, "graphs")

        sub = int(subjects[i].item())
        obj = int(objects[i].item())
        pos_edges.append((sub, obj))  
        sub = int(neg_subjects[i].item())
        obj = int(neg_objects[i].item())
        neg_edges.append((sub, obj))

    # Partitions were already shuffled just take proportionate slice out of array
    # Use 85/5/10% Train/Validation/Test Split
    train_size = int(0.85 * len(pos_edges))
    validation_size = int(0.05 * len(pos_edges))
    test_size = int(0.1 * len(pos_edges))
    pos_train_edges = pos_edges[:train_size]
    neg_train_edges = neg_edges[:train_size]
    pos_validation_edges = pos_edges[train_size:train_size+validation_size]
    neg_validation_edges = neg_edges[train_size:train_size+validation_size]
    pos_test_edges = pos_edges[train_size+validation_size:train_size+validation_size+test_size]
    neg_test_edges = neg_edges[train_size+validation_size:train_size+validation_size+test_size]

    f = open('train_pos.txt', 'w')

    for edge in pos_train_edges:
        f.write(str(edge[0])+'\t'+str(edge[1])+'\n')

    f.close()

    f = open('train_neg.txt', 'w')

    for edge in neg_train_edges:
        f.write(str(edge[0])+'\t'+str(edge[1])+'\n')

    f.close()

    f = open('valid_pos.txt', 'w')

    for edge in pos_validation_edges:
        f.write(str(edge[0])+'\t'+str(edge[1])+'\n')

    f.close()

    f = open('valid_neg.txt', 'w')

    for edge in neg_validation_edges:
        f.write(str(edge[0])+'\t'+str(edge[1])+'\n')

    f.close()

    f = open('test_pos.txt', 'w')

    for edge in pos_test_edges:
        f.write(str(edge[0])+'\t'+str(edge[1])+'\n')

    f.close()

    f = open('test_neg.txt', 'w')

    for edge in neg_test_edges:
        f.write(str(edge[0])+'\t'+str(edge[1])+'\n')

    f.close()


def generate_dataset(dataset_name, whole_graph, feature_key, partition_size, workers, split, multi_type=False):
    if not os.path.isfile(dataset_name+'_partitions.pkl'):
        feature_size = whole_graph.ndata[feature_key].shape[1]
        total_edges = whole_graph.number_of_edges()
        print("Original Graph Edges", total_edges)
        print("Creating", int(whole_graph.number_of_nodes() / partition_size), "simple partitions")

        if not os.path.isfile(dataset_name+'_partition/'+dataset_name+'.json'):
            dgl.distributed.partition_graph(whole_graph, dataset_name, 2, out_path=dataset_name+'_partition', part_method='metis', balance_edges=False, num_trainers_per_machine=workers, objtype='cut')
        if not os.path.isfile(dataset_name+'_simple_partition/'+dataset_name+'_simple_graph.json'):
            dgl.distributed.partition_graph(whole_graph, dataset_name+'_simple_graph', int(whole_graph.number_of_nodes() / partition_size), out_path=dataset_name+'_simple_partition', part_method='metis', balance_edges=False, num_trainers_per_machine=workers, objtype='cut')
        pos_graph_partition = dgl.distributed.load_partition(dataset_name+'_partition/'+dataset_name+'.json', 0, load_feats=True)
        neg_graph_partition = dgl.distributed.load_partition(dataset_name+'_partition/'+dataset_name+'.json', 1, load_feats=True)

        pos_graph = dgl.edge_subgraph(pos_graph_partition[0], pos_graph_partition[0].edata['inner_edge'] == 1)

        if '_N:_E:_N/edge_types' in pos_graph_partition[2].keys():
            pos_graph.edata['edge_types'] = pos_graph_partition[2]['_N:_E:_N/edge_types']

        pos_graph = dgl.node_subgraph(pos_graph, pos_graph.ndata['inner_node'] == 1)
        pos_graph.ndata['feat'] = pos_graph_partition[1]['_N/feat']


        neg_graph = dgl.edge_subgraph(neg_graph_partition[0], neg_graph_partition[0].edata['inner_edge'] == 1)

        if '_N:_E:_N/edge_types' in neg_graph_partition[2].keys():
            neg_graph.edata['edge_types'] = neg_graph_partition[2]['_N:_E:_N/edge_types']

        neg_graph = dgl.node_subgraph(neg_graph, neg_graph.ndata['inner_node'] == 1)
        neg_graph.ndata['feat'] = neg_graph_partition[1]['_N/feat']

        pos_edges = pos_graph.number_of_edges()
        neg_edges = neg_graph.number_of_edges()
        print("Partition 0 Graph Edges", pos_edges)
        print("Partition 1 Graph Edges", neg_edges)
        edges_lost = (total_edges - (pos_edges + neg_edges))
        print("Edges lost", edges_lost, "Percentage of total graph", edges_lost / total_edges)

        if not os.path.isfile(dataset_name+'_pos_partition/'+dataset_name+'_pos_graph.json'):
            dgl.distributed.partition_graph(pos_graph, dataset_name+'_pos_graph', int(pos_graph.number_of_nodes() / partition_size), out_path=dataset_name+'_pos_partition', part_method='metis', balance_edges=False, num_trainers_per_machine=workers, objtype='cut')
            dgl.distributed.partition_graph(neg_graph, dataset_name+'_neg_graph', int(neg_graph.number_of_nodes() / partition_size), out_path=dataset_name+'_neg_partition', part_method='metis', balance_edges=False, num_trainers_per_machine=workers, objtype='cut')
        pos_graphs = []
        neg_graphs = []
        simple_graphs = []
        pos_graphs_count = int(pos_graph.number_of_nodes() / partition_size)
        neg_graphs_count = int(neg_graph.number_of_nodes() / partition_size)
        simple_graphs_count = int(whole_graph.number_of_nodes() / partition_size)

        total_final_edges = 0
        total_edges_per_graph = 0

        for i in range(simple_graphs_count):
            simple_graph_partition = dgl.distributed.load_partition(dataset_name+'_simple_partition/'+dataset_name+'_simple_graph.json', i, load_feats=True)
            simple_graph = dgl.edge_subgraph(simple_graph_partition[0], simple_graph_partition[0].edata['inner_edge'] == 1)

            if '_N:_E:_N/edge_types' in simple_graph_partition[2].keys():
                simple_graph.edata['edge_types'] = simple_graph_partition[2]['_N:_E:_N/edge_types']

            simple_graph = dgl.node_subgraph(simple_graph, simple_graph.ndata['inner_node'] == 1)
            simple_graph.ndata['feat'] = simple_graph_partition[1]['_N/feat']
            
            total_edges_per_graph += simple_graph.number_of_edges()
            simple_graphs.append(simple_graph)

        for i in range(pos_graphs_count):
            pos_graph_partition = dgl.distributed.load_partition(dataset_name+'_pos_partition/'+dataset_name+'_pos_graph.json', i, load_feats=True)
            pos_graph = dgl.edge_subgraph(pos_graph_partition[0], pos_graph_partition[0].edata['inner_edge'] == 1)

            if '_N:_E:_N/edge_types' in pos_graph_partition[2].keys():
                pos_graph.edata['edge_types'] = pos_graph_partition[2]['_N:_E:_N/edge_types']
            
            pos_graph = dgl.node_subgraph(pos_graph, pos_graph.ndata['inner_node'] == 1)
            pos_graph.ndata['feat'] = pos_graph_partition[1]['_N/feat']
            
            total_final_edges += pos_graph.number_of_edges()
            pos_graphs.append(pos_graph)
        
        for i in range(neg_graphs_count):
            neg_graph_partition = dgl.distributed.load_partition(dataset_name+'_neg_partition/'+dataset_name+'_neg_graph.json', i, load_feats=True)
            neg_graph = dgl.edge_subgraph(neg_graph_partition[0], neg_graph_partition[0].edata['inner_edge'] == 1)

            if '_N:_E:_N/edge_types' in neg_graph_partition[2].keys():
                neg_graph.edata['edge_types'] = neg_graph_partition[2]['_N:_E:_N/edge_types']

            neg_graph = dgl.node_subgraph(neg_graph, neg_graph.ndata['inner_node'] == 1)
            neg_graph.ndata['feat'] = neg_graph_partition[1]['_N/feat']
            
            total_final_edges += neg_graph.number_of_edges()
            neg_graphs.append(neg_graph)

        print("Average edges per graph", total_edges_per_graph / len(simple_graphs))
        print("Generated", len(pos_graphs), "positive graphs and", len(neg_graphs), "negative graphs with percentage of original edges remaining", total_final_edges / total_edges)

        graph_output = open(dataset_name+'_partitions.pkl', 'wb')
        random.shuffle(pos_graphs)
        random.shuffle(neg_graphs)
        random.shuffle(simple_graphs)
        
        partitions = {
            "pos_graphs": pos_graphs,
            "neg_graphs": neg_graphs,
            "simple_graphs": simple_graphs
        }

        pickle.dump(partitions, graph_output)
    else:
        partitions_input = open(dataset_name+'_partitions.pkl', 'rb')
        partitions = pickle.load(partitions_input)

    simple_graphs = []

    for i in range(len(partitions["simple_graphs"])):
        simple_graphs.append(partitions["simple_graphs"][i])

    processed_simple_graphs = []
    processed_graphs_true = []
    processed_graphs_25_edge = []
    processed_graphs_50_edge = []
    processed_graphs_75_edge = []
    processed_graphs_90_edge = []
    processed_graphs_25_node = []
    processed_graphs_50_node = []
    processed_graphs_75_node = []
    processed_graphs_90_node = []

    for graph in simple_graphs:
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
        graph_dict_true['gnn_feature'] = graph.ndata['feat']
        graph_dict_25_edge['gnn_feature'] = graph.ndata['feat']
        graph_dict_50_edge['gnn_feature'] = graph.ndata['feat']
        graph_dict_75_edge['gnn_feature'] = graph.ndata['feat']
        graph_dict_90_edge['gnn_feature'] = graph.ndata['feat']
        graph_dict_25_node['gnn_feature'] = graph.ndata['feat']
        graph_dict_50_node['gnn_feature'] = graph.ndata['feat']
        graph_dict_75_node['gnn_feature'] = graph.ndata['feat']
        graph_dict_90_node['gnn_feature'] = graph.ndata['feat']

        if 'edge_types' in graph.edata.keys():
            graph_dict['edge_feature'] = graph.edata['edge_types']
            graph_dict_true['edge_feature'] = graph.edata['edge_types']
            graph_dict_25_edge['edge_feature'] = graph.edata['edge_types']
            graph_dict_50_edge['edge_feature'] = graph.edata['edge_types']
            graph_dict_75_edge['edge_feature'] = graph.edata['edge_types']
            graph_dict_90_edge['edge_feature'] = graph.edata['edge_types']
            graph_dict_25_node['edge_feature'] = graph.edata['edge_types']
            graph_dict_50_node['edge_feature'] = graph.edata['edge_types']
            graph_dict_75_node['edge_feature'] = graph.edata['edge_types']
            graph_dict_90_node['edge_feature'] = graph.edata['edge_types']

        pos_edges = []
        neg_edges = []
        pos_types = []
        neg_types = []

        total_pos_edges = len(graph.edges()[0])
        negative_graph_edges = dgl.sampling.global_uniform_negative_sampling(graph, total_pos_edges)
        total_neg_edges = len(negative_graph_edges[0])

        for i in range(min([total_pos_edges, total_neg_edges])):
            sub = int(graph.edges()[0][i].item())
            obj = int(graph.edges()[1][i].item())
            pos_edges.append((sub, obj))
            if 'edge_types' in graph.edata.keys():
                pos_types.append(graph.edata['edge_types'][i].numpy().item())  
            sub = int(negative_graph_edges[0][i].item())
            obj = int(negative_graph_edges[1][i].item())
            neg_edges.append((sub, obj))
            if 'edge_types' in graph.edata.keys():
                neg_types.append(random.randint(0, max(graph.edata['edge_types'])))  

        graph_dict['pos_edges'] = pos_edges
        graph_dict['neg_edges'] = neg_edges
        graph_dict['pos_types'] = pos_types
        graph_dict['neg_types'] = neg_types
        graph_dict['node_count'] = graph.number_of_nodes()
        processed_simple_graphs.append(graph_dict)

        graph_dict_true['pos_edges'] = pos_edges
        graph_dict_true['given_edges'] = []
        graph_dict_true['neg_edges'] = neg_edges
        graph_dict_true['node_count'] = graph.number_of_nodes()
        graph_dict_true['pos_types'] = pos_types
        graph_dict_true['neg_types'] = neg_types
        processed_graphs_true.append(graph_dict_true)

        random.shuffle(pos_edges)
        random.shuffle(neg_edges)

        edges_retained_25 = max([1, int(len(pos_edges) * 0.25)])
        edges_retained_50 = max([1, int(len(pos_edges) * 0.50)])
        edges_retained_75 = max([1, int(len(pos_edges) * 0.75)])
        edges_retained_90 = max([1, int(len(pos_edges) * 0.90)])
        
        graph_dict_25_edge['pos_edges'] = pos_edges[edges_retained_25:]
        graph_dict_25_edge['given_edges'] = pos_edges[:edges_retained_25]
        graph_dict_25_edge['neg_edges'] = neg_edges[edges_retained_25:]
        graph_dict_25_edge['node_count'] = graph.number_of_nodes()
        graph_dict_25_edge['pos_types'] = pos_types[edges_retained_25:]
        graph_dict_25_edge['given_types'] = pos_types[:edges_retained_25]
        graph_dict_25_edge['neg_types'] = neg_types[edges_retained_25:]
        processed_graphs_25_edge.append(graph_dict_25_edge)

        graph_dict_50_edge['pos_edges'] = pos_edges[edges_retained_50:]
        graph_dict_50_edge['given_edges'] = pos_edges[:edges_retained_50]
        graph_dict_50_edge['neg_edges'] = neg_edges[edges_retained_50:]
        graph_dict_50_edge['node_count'] = graph.number_of_nodes()
        graph_dict_50_edge['pos_types'] = pos_types[edges_retained_50:]
        graph_dict_50_edge['given_types'] = pos_types[:edges_retained_50]
        graph_dict_50_edge['neg_types'] = neg_types[edges_retained_50:]
        processed_graphs_50_edge.append(graph_dict_50_edge)

        graph_dict_75_edge['pos_edges'] = pos_edges[edges_retained_75:]
        graph_dict_75_edge['given_edges'] = pos_edges[:edges_retained_75]
        graph_dict_75_edge['neg_edges'] = neg_edges[edges_retained_75:]
        graph_dict_75_edge['node_count'] = graph.number_of_nodes()
        graph_dict_75_edge['pos_types'] = pos_types[edges_retained_75:]
        graph_dict_75_edge['given_types'] = pos_types[:edges_retained_75]
        graph_dict_75_edge['neg_types'] = neg_types[edges_retained_75:]
        processed_graphs_75_edge.append(graph_dict_75_edge)

        graph_dict_90_edge['pos_edges'] = pos_edges[edges_retained_90:]
        graph_dict_90_edge['given_edges'] = pos_edges[:edges_retained_90]
        graph_dict_90_edge['neg_edges'] = neg_edges[edges_retained_90:]
        graph_dict_90_edge['node_count'] = graph.number_of_nodes()
        graph_dict_90_edge['pos_types'] = pos_types[edges_retained_90:]
        graph_dict_90_edge['given_types'] = pos_types[:edges_retained_90]
        graph_dict_90_edge['neg_types'] = neg_types[edges_retained_90:]
        processed_graphs_90_edge.append(graph_dict_90_edge)

        num_nodes = graph.number_of_nodes()
        nodes_retained_25 = max([1, int(num_nodes * 0.25)])
        nodes_to_remove_25 = random.sample(range(num_nodes), (num_nodes - nodes_retained_25))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for i in range(len(pos_edges)):
            edge = pos_edges[i]
            if (edge[0] not in nodes_to_remove_25) and (edge[1] not in nodes_to_remove_25):
                given_edges.append(edge)
                if len(pos_types):
                    given_types.append(pos_types[i])
            else:
                positive_edges.append(edge)
                if len(pos_types):
                    positive_types.append(pos_types[i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_25_node['pos_edges'] = positive_edges
        graph_dict_25_node['given_edges'] = given_edges
        graph_dict_25_node['neg_edges'] = negative_edges
        graph_dict_25_node['pos_types'] = positive_types
        graph_dict_25_node['given_types'] = given_types
        graph_dict_25_node['node_count'] = graph.number_of_nodes()
        processed_graphs_25_node.append(graph_dict_25_node)

        nodes_retained_50 = max([1, int(num_nodes * 0.50)])
        nodes_to_remove_50 = random.sample(range(num_nodes), (num_nodes - nodes_retained_50))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for i in range(len(pos_edges)):
            edge = pos_edges[i]
            if (edge[0] not in nodes_to_remove_50) and (edge[1] not in nodes_to_remove_50):
                given_edges.append(edge)
                if len(pos_types):
                    given_types.append(pos_types[i])
            else:
                positive_edges.append(edge)
                if len(pos_types):
                    positive_types.append(pos_types[i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_50_node['pos_edges'] = positive_edges
        graph_dict_50_node['given_edges'] = given_edges
        graph_dict_50_node['neg_edges'] = negative_edges
        graph_dict_50_node['pos_types'] = positive_types
        graph_dict_50_node['given_types'] = given_types
        graph_dict_50_node['node_count'] = graph.number_of_nodes()
        processed_graphs_50_node.append(graph_dict_50_node)

        nodes_retained_75 = max([1, int(num_nodes * 0.75)])
        nodes_to_remove_75 = random.sample(range(num_nodes), (num_nodes - nodes_retained_75))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for i in range(len(pos_edges)):
            edge = pos_edges[i]
            if (edge[0] not in nodes_to_remove_75) and (edge[1] not in nodes_to_remove_75):
                given_edges.append(edge)
                if len(pos_types):
                    given_types.append(pos_types[i])
            else:
                positive_edges.append(edge)
                if len(pos_types):
                    positive_types.append(pos_types[i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_75_node['pos_edges'] = positive_edges
        graph_dict_75_node['given_edges'] = given_edges
        graph_dict_75_node['neg_edges'] = negative_edges
        graph_dict_75_node['pos_types'] = positive_types
        graph_dict_75_node['given_types'] = given_types
        graph_dict_75_node['node_count'] = graph.number_of_nodes()
        processed_graphs_75_node.append(graph_dict_75_node)

        nodes_retained_90 = max([1, int(num_nodes * 0.90)])
        nodes_to_remove_90 = random.sample(range(num_nodes), (num_nodes - nodes_retained_90))
        given_edges = []
        positive_edges = []
        given_types = []
        positive_types = []

        for i in range(len(pos_edges)):
            edge = pos_edges[i]
            if (edge[0] not in nodes_to_remove_90) and (edge[1] not in nodes_to_remove_90):
                given_edges.append(edge)
                if len(pos_types):
                    given_types.append(pos_types[i])
            else:
                positive_edges.append(edge)
                if len(pos_types):
                    positive_types.append(pos_types[i])

            negative_edges = neg_edges[:len(positive_edges)]

        graph_dict_90_node['pos_edges'] = positive_edges
        graph_dict_90_node['given_edges'] = given_edges
        graph_dict_90_node['neg_edges'] = negative_edges
        graph_dict_90_node['pos_types'] = positive_types
        graph_dict_90_node['given_types'] = given_types
        graph_dict_90_node['node_count'] = graph.number_of_nodes()
        processed_graphs_90_node.append(graph_dict_90_node)

    # Partitions were already shuffled just take proportionate slice out of array
    # Use 85/5/10% Train/Validation/Test Split
    train_size = int(split[0] * len(simple_graphs))
    validation_size = int(split[1] * len(simple_graphs))
    test_size = int(split[2] * len(simple_graphs))
    simple_train_graphs = processed_simple_graphs[:train_size]
    simple_validation_graphs = processed_simple_graphs[train_size:train_size+validation_size]
    simple_test_graphs = processed_simple_graphs[train_size+validation_size:train_size+validation_size+test_size]

    simple_graphs = {
        'train': simple_train_graphs,
        'valid': simple_validation_graphs,
        'test': simple_test_graphs
    }

    simple_file = open(dataset_name+'_simple_graphs.pkl', 'wb')
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