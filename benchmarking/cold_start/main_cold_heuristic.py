import torch
import time
import numpy as np
import argparse
import scipy.sparse as ssp
from collections import Counter
import pickle
import sys
sys.path.append("..") 

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from utils import *
from get_heuristic import *
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc


dir_path = get_root_dir()

def read_data(data_name, dir_path, cold_perc, blind):
    if cold_perc > 0.0:
        if cold_perc == 0.25:
            cold_part = '25'
        if cold_perc == 0.50:
            cold_part = '50'
        if cold_perc == 0.75:
            cold_part = '75'
        if cold_perc == 0.90:
            cold_part = '90'

        path = dir_path+ '/{}/{}_{}_graphs.pkl'.format(data_name, data_name, cold_part+'_'+blind)
    else:
        path = dir_path+ '/{}/{}_{}_graphs.pkl'.format(data_name, data_name, 'true')
    graphs_input = open(path, 'rb')
    graphs = pickle.load(graphs_input)
    data = {
        'train': [],
        'valid': [],
        'test': []
    }
    
    for graphs_key in graphs.keys():
        for graph in graphs[graphs_key]:
            train_pos = graph['pos_edges']
            train_neg = graph['neg_edges']
            given_edges = graph['given_edges']
            num_nodes = graph['node_count']
            positive_edges = train_pos
            negative_edges = train_neg

            if (len(train_pos) > 0) and (len(train_neg) > 0):
                if (cold_perc == 0.0) or (len(given_edges) == 0):
                    adj = ssp.csr_matrix(torch.zeros((num_nodes, num_nodes)), shape=(num_nodes, num_nodes))
                else:
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)

                    if 'pos_types' in graph.keys():
                        edge_index = adj_edge
                        edge_weight = torch.tensor(graph['given_types'], dtype=torch.float32)
                    else:
                        edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                        edge_weight = torch.ones(edge_index.size(1))

                    adj = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 
                    
                graph_data = {}
                graph_data['adj'] = adj
                graph_data['pos'] = torch.transpose(torch.tensor(positive_edges), 1, 0)
                graph_data['neg'] = torch.transpose(torch.tensor(negative_edges), 1, 0)
                graph_data['x'] = graph['gnn_feature']
                graph_data['node_count'] = graph['node_count']
                data[graphs_key].append(graph_data)

    train_valid_count = len(data['valid'])
    data['train_valid'] = data['train'][:train_valid_count]

    return data

def get_average_results(train, valid, test):
    all_result = {}
    train_total = 0.0
    valid_total = 0.0
    test_total = 0.0
    result_mrr_train = {'MRR': 0.0}
    result_mrr_valid = {'MRR': 0.0}
    result_mrr_test = {'MRR': 0.0}

    for K in [1,3,10, 100]:
        result_mrr_train[f'Hits@{K}'] = 0.0
        result_mrr_valid[f'Hits@{K}'] = 0.0
        result_mrr_test[f'Hits@{K}'] = 0.0

    for result in train:
        train_total += 1.0
        result_mrr_train['MRR'] += result[1]['MRR']

        for K in [1,3,10, 100]:
            result_mrr_train[f'Hits@{K}'] += result[0][f'Hits@{K}']

    for result in valid:
        valid_total += 1.0
        result_mrr_valid['MRR'] += result[1]['MRR']

        for K in [1,3,10, 100]:
            result_mrr_valid[f'Hits@{K}'] += result[0][f'Hits@{K}']

    for result in test:
        test_total += 1.0
        result_mrr_test['MRR'] += result[1]['MRR']

        for K in [1,3,10, 100]:
            result_mrr_test[f'Hits@{K}'] += result[0][f'Hits@{K}']

    result_mrr_train['MRR'] = result_mrr_train['MRR'] / train_total
    result_mrr_valid['MRR'] = result_mrr_valid['MRR'] / valid_total
    result_mrr_test['MRR'] = result_mrr_test['MRR'] / test_total

    for K in [1,3,10, 100]:
        result_mrr_train[f'Hits@{K}'] = result_mrr_train[f'Hits@{K}'] / train_total
        result_mrr_valid[f'Hits@{K}'] = result_mrr_valid[f'Hits@{K}'] / valid_total
        result_mrr_test[f'Hits@{K}'] = result_mrr_test[f'Hits@{K}'] / test_total

    all_result['MRR'] = (result_mrr_train['MRR'], result_mrr_valid['MRR'], result_mrr_test['MRR'])
    for K in [1,3,10, 100]:
        all_result[f'Hits@{K}'] = (result_mrr_train[f'Hits@{K}'], result_mrr_valid[f'Hits@{K}'], result_mrr_test[f'Hits@{K}'])
    
    return all_result

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='katz_apro')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--beta', type=float, default='0.005')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')

    args = parser.parse_args()

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind)

    use_heuristic = args.use_heuristic
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    train_results = []
    valid_results = []
    test_results = []
    train_memory = []
    test_memory = []
    train_times = []
    test_times = []


    start_time = time.time()
    for graph in data['train_valid']:
        pos_pred = eval(use_heuristic)(graph['adj'], graph['pos'])
        neg_pred = eval(use_heuristic)(graph['adj'], graph['neg'])
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))
        train_memory.append(torch.cuda.max_memory_allocated(device=None))
    train_times.append(time.time() - start_time)

    for graph in data['valid']:
        pos_pred = eval(use_heuristic)(graph['adj'], graph['pos'])
        neg_pred = eval(use_heuristic)(graph['adj'], graph['neg'])
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    start_time = time.time()
    for graph in data['test']:
        pos_pred = eval(use_heuristic)(graph['adj'], graph['pos'])
        neg_pred = eval(use_heuristic)(graph['adj'], graph['neg'])
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        test_results.append((hits, mrr))
        test_memory.append(torch.cuda.max_memory_allocated(device=None))
    test_times.append(time.time() - start_time)

    result = get_average_results(train_results, valid_results, test_results)

    print('heurisitic: ', args.use_heuristic)    
    print('train:  mrr of ' + args.data_name + ' is: ', 100*result['MRR'][0])
    print('valid:  mrr of ' + args.data_name + ' is: ', 100*result['MRR'][1])
    print('test: mrr of ' + args.data_name + ' is: ', 100*result['MRR'][2])

    print('test: hit 1, 3, 10, 100 of ' + args.data_name + ' is: ', 100*result['Hits@1'][2], 100*result['Hits@3'][2], 100*result['Hits@10'][2], 100*result['Hits@100'][2])
    
    print("Training max memory:", max(train_memory))
    print("Testing max memory:", max(test_memory))
    print("Average total train time (s)", sum(train_times))
    print("Average total test time (s)", sum(test_times))

    print("Training run times:")
    for i in range(len(train_times)):
        print("\tTrain time for run", i+1, train_times[i])

    print("Testing run times:")
    for i in range(len(test_times)):
        print("Testing time for run", i+1, test_times[i])


if __name__ == "__main__":
    main()