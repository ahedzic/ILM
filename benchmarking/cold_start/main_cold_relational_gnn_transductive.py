
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from torch_geometric.nn import TransE, RotatE, ComplEx, DistMult
from utils import *
import random
import time
import statistics
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

log_print		= get_logger('testrun', 'log', get_config_dir())
def read_data(data_name, dir_path, cold_perc, blind, device):
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
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), value=torch.empty(0, dtype=torch.float32), sparse_sizes=[num_nodes, num_nodes])
                    edge_weight = torch.empty(0, dtype=torch.long)
                else:
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)

                    if 'pos_types' in graph.keys():
                        edge_index = adj_edge

                        if len(graph['given_types']):
                            edge_weight = torch.tensor(graph['given_types'], dtype=torch.long)
                        else:
                            edge_weight = torch.empty(0, dtype=torch.long)
                    else:
                        edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                        edge_weight = torch.ones(edge_index.size(1))

                    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

                train_size = int(0.85 * len(pos_edges))
                test_size = int(0.15 * len(pos_edges))
                pos_train_edges = pos_edges[:train_size]
                neg_train_edges = neg_edges[:train_size]
                pos_test_edges = pos_edges[train_size:train_size+test_size]
                neg_test_edges = neg_edges[train_size:train_size+test_size]

                graph_data = {}
                graph_data['adj'] = adj
                graph_data['adj_type'] = edge_weight
                graph_data['pos'] = torch.transpose(torch.tensor(pos_train_edges), 1, 0)
                graph_data['neg'] = torch.transpose(torch.tensor(neg_train_edges), 1, 0)
                graph_data['pos_test'] = torch.transpose(torch.tensor(pos_test_edges), 1, 0)
                graph_data['neg_test'] = torch.transpose(torch.tensor(neg_test_edges), 1, 0)
                graph_data['x'] = graph['gnn_feature']
                graph_data['node_count'] = graph['node_count']

                if len(pos_test_edges) > 0:
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
        
@torch.no_grad()
def test_edge(score_func, graph, edges, model, device):
    pos_preds = []
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    edge_types = graph['adj_type'].to(device)

    h = model(x, adj, edge_types)
    edges = edges
    pos_scores = score_func(h[edges[0]], h[edges[1]]).cpu()
    pos_preds += [pos_scores]
          
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds


@torch.no_grad()
def test(model, score_func, graph, evaluator_hit, evaluator_mrr, device):
    pos_pred = test_edge(score_func, graph['pos_test'], model, device)
    neg_pred = test_edge(score_func, graph['neg_test'], model, device)
    pos_pred = torch.flatten(pos_pred)
    neg_pred = torch.flatten(neg_pred)
    k_list = [1, 3, 10, 100]
    hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
    mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))

    return (hits, mrr)


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--model', type=str, default='TransE')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')
    parser.add_argument('--num_relations', type=int, default=1)

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
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

    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind, args.device)

    #model = eval(args.model)(data['n_total'], args.rank, args.alpha, args.beta, args.groups, device).to(device)
    
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

        best_valid = 0
        kill_cnt = 0

        for epoch in range(0, 1):
            total_training_time = 0.0
            total_testing_time = 0.0
            test_results = []

            for graph in data['test']:
                model = eval(args.model)(graph['node_count'], args.num_relations, args.hidden_channels).to(device)
                score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels, 1, args.num_layers_predictor, args.dropout).to(device)
                optimizer = torch.optim.Adam(list(model.parameters()) + list(score_func.parameters()), lr=args.lr, weight_decay=args.l2)
                start_time = time.time()
                training = True
                best = float('inf')

                while training:
                    expected_pos = graph['pos'].to(device)
                    expected_neg = graph['neg'].to(device)
                    x = graph['x'].to(device)
                    adj = graph['adj'].to(device)
                    edge_types = graph['adj_type'].to(device)

                    optimizer.zero_grad()
                    h = model(x, adj, edge_types)
                    edge = expected_pos
                    pos_out = score_func(h[edge[0]], h[edge[1]])
                    pos_loss = -torch.log(pos_out + 1e-15).mean()

                    edge = expected_neg
                    neg_out = score_func(h[edge[0]], h[edge[1]])
                    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

                    loss = pos_loss + neg_loss
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

                    optimizer.step()
                    best_current = loss.item()

                    if best_current < best:
                        best = best_current
                        kill_cnt = 0
                    else:
                        kill_cnt += 1

                    if kill_cnt > args.kill_cnt:
                        training = False

                train_memory.append(torch.cuda.max_memory_allocated(device=None))
                total_training_time += time.time() - start_time
                start_time = time.time()
                result = test(model, score_func, graph, evaluator_hit, evaluator_mrr, device)
                test_results.append(result)
                test_memory.append(torch.cuda.max_memory_allocated(device=None))
                total_testing_time += time.time() - start_time

            train_times.append(total_training_time)
            test_times.append(total_testing_time)
            results_rank = get_average_results(test_results, test_results, test_results)
            
            if epoch % args.eval_steps == 0:
                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                       

                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {(0.0):.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                
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

    if args.model != 'Leroy':
        print("Training max memory (bytes):", max(train_memory))
        print("Testing max memory (bytes):", max(test_memory))

        if len(train_times) > 1:
            print("Training run time per epoch (s)", statistics.mean(train_times), "+-", statistics.stdev(train_times))
            print("Testing run times per epoch (s)", statistics.mean(test_times), "+-", statistics.stdev(test_times))
        else:
            print("Training run time per epoch (s)", train_times[0])
            print("Testing run times per epoch (s)", test_times[0])
    else:
        print("Testing max memory (bytes):", max(test_memory))
        print("Testing run times per epoch (s)", test_times[0])



    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   