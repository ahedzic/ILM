
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from deal_model import *
from utils import *
import random
import networkx as nx
import math
import multiprocessing as mp
import time
import statistics
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import assortativity, to_networkx
from torch_geometric.data import Data


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from argparse import ArgumentParser

clust = []
isolated = []
mags = []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Data:
    x = None
    dists = None
    def __init__(self, x, dists):
        self.x = x
        self.dists = dists
    def copy(self):
        return Data(self.x, self.dists)

log_print		= get_logger('testrun', 'log', get_config_dir())
def read_data(data_name, dir_path, cold_perc, blind, num_relations):
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

    offsets = {}
    features = []
    total_nodes = 0
    total_test_nodes = 0
    graph_index = 0
    graph_data = {}

    for graphs_key in graphs.keys():
        graph_data[graphs_key] = []
        for graph in graphs[graphs_key]:
            offsets[graph_index] = total_nodes
            total_nodes += graph['node_count']
            features.append(graph['gnn_feature'])
            graph_index += 1

            if graphs_key == 'test':
                total_test_nodes += graph['node_count']

    global_index = None
    global_edge_weight = None
    global_train_index = None
    global_train_edge_weight = None
    train_node_indices = []
    test_node_indices = []
    graph_index = 0
    test_pos_edges = []
    test_neg_edges = []
    train_pos_edges = []
    train_neg_edges = []
    
    for graphs_key in graphs.keys():
        for graph in graphs[graphs_key]:
            train_pos = graph['pos_edges']
            train_neg = graph['neg_edges']
            given_edges = graph['given_edges']
            num_nodes = graph['node_count']
            positive_edges = train_pos
            negative_edges = train_neg
            graph_map = {}

            adj_pos_edge = torch.transpose(torch.tensor(train_pos), 1, 0)

            if cold_perc > 0.0:
                given_types = torch.tensor(graph['given_types'], dtype=torch.long)
                adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)
                adj_edge = torch.cat((adj_edge,  adj_edge[[1,0]], adj_pos_edge,  adj_pos_edge[[1,0]]), dim=1)

                if 'pos_types' in graphs.keys() and len(graph['pos_types']):
                    pos_types = torch.tensor(graph['pos_types'], dtype=torch.long)
                    edge_weights = torch.cat((given_types, given_types, pos_types, pos_types))
                else:
                    edge_weights = torch.ones(adj_edge.size(1))

                adj_train_edge = torch.transpose(torch.tensor(given_edges), 1, 0)
                adj_train_edge = torch.cat((adj_train_edge,  adj_train_edge[[1,0]]), dim=1)
                train_edge_weights = torch.cat((given_types, given_types))
            else:
                adj_edge = torch.cat((adj_pos_edge,  adj_pos_edge[[1,0]]), dim=1)

                if 'pos_types' in graphs.keys() and len(graph['pos_types']):
                    pos_types = torch.tensor(graph['pos_types'], dtype=torch.long)
                    edge_weights = torch.cat((pos_types, pos_types))
                else:
                    edge_weights = torch.ones(adj_edge.size(1))

                adj_train_edge = adj_edge
                train_edge_weights = edge_weights

            offset = offsets[graph_index]
            adj_edge = adj_edge + offset
            adj_train_edge = adj_train_edge + offset

            if graph_index == 0:
                global_index = adj_edge
                global_edge_weight = edge_weights
                global_train_index = adj_train_edge
                global_train_edge_weight = train_edge_weights
            else:
                global_index = torch.cat((global_index, adj_edge), dim=1)
                global_edge_weight = torch.cat((global_edge_weight, edge_weights))
                global_train_index = torch.cat((global_train_index, adj_train_edge), dim=1)
                global_train_edge_weight = torch.cat((global_train_edge_weight, train_edge_weights))

            pos_edges = torch.tensor(train_pos) + offset
            neg_edges = torch.tensor(train_neg) + offset

            graph_map["pos"] = pos_edges
            graph_map["neg"] = neg_edges
            graph_map["inputs"] = torch.cat((pos_edges, neg_edges), dim=0)
            graph_map["labels"] = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))))

            graph_index += 1
            graph_data[graphs_key].append(graph_map)
                    
    graph_data['edge_index'] = global_index
    graph_data['data'] = Data(torch.cat(features, dim=0), precompute_dist_data(global_index, total_nodes))
    graph_data['node_count'] = total_nodes
    train_valid_count = len(graph_data['valid'])
    graph_data['train_valid'] = graph_data['train'][:train_valid_count]

    return graph_data

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array


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
        

def train(model, graph, optimizer, data, device):
    labels = graph["labels"].to(device)
    inputs = graph["inputs"].to(device)
    optimizer.zero_grad()
    loss = model.default_loss(inputs, labels, data, thetas=(0.1,0.85,0.05))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_edge(graph, pos_edges, neg_edges, model, data, device):
    pos_preds = []
    neg_preds = []

    res = model.evaluate(pos_edges, data, (0.1,0.85,0.05))

    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]

    pos_scores = res.detach().cpu()

    res = model.evaluate(neg_edges, data, (0.1,0.85,0.05))
    
    if len(res.shape)>1:
        res = res.softmax(dim=1)[:,1]

    neg_scores = res.detach().cpu()
    
    pos_preds += [pos_scores]
    neg_preds += [neg_scores]
    
    if len(pos_preds) > 1:
        pos_preds = torch.cat(pos_preds, dim=0)
    else:
        pos_preds = pos_preds[0]

    if len(neg_preds) > 1:
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = neg_preds[0]

    return pos_preds, neg_preds


@torch.no_grad()
def test(model, evaluator_hit, evaluator_mrr, data, device):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred, neg_pred = test_edge(graph, graph['pos'], graph['neg'], model, data["data"], device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred, neg_pred = test_edge(graph, graph['pos'], graph['neg'], model, data["data"], device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred, neg_pred = test_edge(graph, graph['pos'], graph['neg'], model, data["data"], device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        test_results.append((hits, mrr))
    
    result = get_average_results(train_results, valid_results, test_results)
    
    score_emb = [pos_pred.cpu(),neg_pred.cpu(), pos_pred.cpu(), neg_pred.cpu()]

    return result, score_emb


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='starcraft')
    parser.add_argument('--input_size', type=int, default=602)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='RGCN')
    parser.add_argument('--score_model', type=str, default='meta_score')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')
    parser.add_argument('--max_nodes', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--num_relations', type=int, default=1)
    parser.add_argument('--weights', type=str, default='meta')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--meta_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    
    # state = torch.load('output_test/lr0.01_drop0.1_l20.0001_numlayer1_numPredlay2_numGinMlplayer2_dim64_best_run_0')

    #### 
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')

    parser.add_argument('--mode', dest='train_mode', default='cos', type=str,
                        help='cos, dot, all, pdist, default cos')

    parser.add_argument('--attr_model', dest='attr_model', default='Emb', type=str,
                        help='Attribute embedding model, Emb, SAGE, GAT ... , default Emb')

    parser.add_argument('--bce', dest='BCE_mode', default=True, type=str2bool, 
                help='If use BCE_mode, default True')

    parser.add_argument('--sa', dest='strong_A', action='store_true',
                        help='use Strong Alignment')

    parser.add_argument('--gamma', dest='gamma', default=2, type=float)

    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind, args.num_relations)
    data['data'].x = data['data'].x.to(device)
    data['data'].dists = torch.Tensor(data['data'].dists).to(device)
    features = data['data'].x
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
       
    }

    train_memory = []
    test_memory = []
    train_times = []
    test_times = []

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        deal = DEAL(args.hidden_channels, features.shape[1], features.shape[0], device, args, Emb)
        optimizer = torch.optim.Adam(deal.parameters(), lr=args.lr) 

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            deal.train()
            loss = 0.0
            loss_count = 0
            
            start_time = time.time()
            for graph in data['train']:
                loss += train(deal, graph, optimizer, data['data'], device)
                loss_count +=1
                train_memory.append(torch.cuda.max_memory_allocated(device=None))
            train_times.append(time.time() - start_time)
            
            if epoch % args.eval_steps == 0:
                deal.eval()
                start_time = time.time()
                results_rank, score_emb = test(deal, evaluator_hit, evaluator_mrr, data, device)
                test_memory.append(torch.cuda.max_memory_allocated(device=None))
                test_times.append(time.time() - start_time)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                       

                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {(loss / loss_count):.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:

                        save_emb(score_emb, save_path)

                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            
            print(key)
            loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():

        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean


            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    # print(best_metric_valid_str +' ' +best_auc_valid_str)

    print(best_metric_valid_str)
    best_auc_metric = best_valid_mean_metric

    print("Training max memory (bytes):", max(train_memory))
    print("Testing max memory (bytes):", max(test_memory))
    print("Average total train time (s)", sum(train_times) / float(args.runs))
    print("Average total test time (s)", sum(test_times) / float(args.runs))
    print("Training run time per epoch (s)", statistics.mean(train_times), "+-", statistics.stdev(train_times))
    print("Testing run times per epoch (s)", statistics.mean(test_times), "+-", statistics.stdev(test_times))

    global clust
    global isolated
    global mags

    with open(args.data_name + 'output_mags.csv', 'w') as file:
        for row in mags:
            file.write(str(row) + ',\n')

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   
