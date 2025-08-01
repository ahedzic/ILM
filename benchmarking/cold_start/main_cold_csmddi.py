
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from csmddi_model import *
from utils import *
import random
import time
import statistics
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

from random import sample
import sklearn.metrics as metrics



log_print		= get_logger('testrun', 'log', get_config_dir())
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

    offsets = {}
    features = []
    total_nodes = 0
    total_test_nodes = 0
    graph_index = 0

    for graphs_key in graphs.keys():
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
    
    for graphs_key in graphs.keys():
        for graph in graphs[graphs_key]:
            train_pos = graph['pos_edges']
            train_neg = graph['neg_edges']
            given_edges = graph['given_edges']
            num_nodes = graph['node_count']
            positive_edges = train_pos
            negative_edges = train_neg

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

            if graphs_key == 'test':
                test_pos_edges.append(torch.tensor(train_pos) + offset)
                test_neg_edges.append(torch.tensor(train_neg) + offset)

                for i in range(offset, offset + num_nodes):
                    test_node_indices.append(i)
                
            else:
                for i in range(offset, offset + num_nodes):
                    train_node_indices.append(i)

            graph_index += 1
            
    adj = SparseTensor.from_edge_index(global_index, global_edge_weight, [total_nodes, total_nodes])
    #adj_train = SparseTensor.from_edge_index(global_train_index, global_train_edge_weight, [total_nodes, total_nodes])
                    
    graph_data = {}
    graph_data['adj'] = adj.to_dense().numpy()
    #graph_data['adj_train'] = adj_train.to_dense().numpy()
    #graph_data['adj_type'] = global_edge_weight
    graph_data['train_nodes'] = train_node_indices
    graph_data['test_nodes'] = test_node_indices
    graph_data['test_pos_edges'] = torch.cat(test_pos_edges, dim=0)
    graph_data['test_neg_edges'] = torch.cat(test_neg_edges, dim=0)
    graph_data['x'] = torch.cat(features, dim=0)
    graph_data['node_count'] = total_nodes

    return graph_data


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

def binary_evaluation_result(label_true, score_predict):
    auc = metrics.roc_auc_score(label_true, score_predict)
    precision, recall, _ = metrics.precision_recall_curve(
        label_true, score_predict)
    aupr = metrics.auc(recall, precision)

    score_sigmoid = 1/(1 + np.exp(-np.array(score_predict)))
    label_predict = [1 if score > 0.5 else 0 for score in score_sigmoid]

    acc = metrics.accuracy_score(label_true, label_predict)
    precision = metrics.precision_score(label_true, label_predict)
    recall = metrics.recall_score(label_true, label_predict)
    f1 = metrics.f1_score(label_true, label_predict)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr,
    }

@torch.no_grad()
def test(model, data, evaluator_hit, evaluator_mrr, device):
    test_results = []
    labels, results = model.test()

    pos_pred = []
    neg_pred = []

    test_indices = {}

    for i in range(len(data['test_nodes'])):
        test_indices[data['test_nodes'][i]] = i

    pos_edges = data['test_pos_edges']
    neg_edges = data['test_neg_edges']

    for i in range(len(data['test_pos_edges'])):
        pos_pred.append(results[test_indices[pos_edges[i][0].item()], test_indices[pos_edges[i][1].item()]])
    for i in range(len(data['test_neg_edges'])):
        neg_pred.append(results[test_indices[neg_edges[i][0].item()], test_indices[neg_edges[i][1].item()]])

    pos_pred = torch.tensor(pos_pred)
    neg_pred = torch.tensor(neg_pred)
  
    pos_pred = torch.flatten(pos_pred)
    neg_pred = torch.flatten(neg_pred)
    k_list = [1, 3, 10, 100]
    hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
    mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
    test_results.append((hits, mrr))
    
    result = get_average_results(test_results, test_results, test_results)

    return result


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--input_size', type=int, default=602)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')
    parser.add_argument('--num_relations', type=int, default=1)

    ##gnn setting
    parser.add_argument('--hidden_channels', type=int, default=64)


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

    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind)

    input_channel = args.input_size

    config = {}
    config['device'] = device
    config['epoch_num'] = 10
    config['binary_or_multi'] = 'binary'
    config['S1_or_S2'] = 'S2'
    config['relation_learning_model'] = 'TransE'
    config['map_model'] = 'PLSR'
    config['drug_hidden_embedding_dim'] = args.hidden_channels
    config['batch_size'] = 200
    config['learning_rate'] = args.lr
    config['epoch_num'] = 10000
    config['cv'] = 10
    config['shuffle_drug'] = False

    data['interaction_num'] = args.num_relations

    if args.num_relations > 1:
        config['adj'] = 'adj_multi'
    else:
        config['adj'] = 'adj'
    
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

        best_valid = 0
        kill_cnt = 0
        loss = 0.0
        loss_count = 0
        model = ColdStartRescalTensorFactorizationTorch(config, data)
            
        start_time = time.time()
        model.init(data['train_nodes'], data['test_nodes'])
        model.train()
        train_memory.append(torch.cuda.max_memory_allocated(device=None))
        train_times.append(time.time() - start_time)
        start_time = time.time()
        results_rank = test(model, data, evaluator_hit, evaluator_mrr, device)
        test_memory.append(torch.cuda.max_memory_allocated(device=None))
        test_times.append(time.time() - start_time)

        for key, result in results_rank.items():
            loggers[key].add_result(run, result)

            for key, result in results_rank.items():
                        
                print(key)
                        
                train_hits, valid_hits, test_hits = result
                       

                log_print.info(
                    f'Run: {run + 1:02d}, '
                    f'Test: {100 * test_hits:.2f}%')
            print('---')
        
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


    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   