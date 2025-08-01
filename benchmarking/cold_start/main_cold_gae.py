import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score
import random
import time
import statistics
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

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
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), sparse_sizes=[num_nodes, num_nodes])
                else:
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)

                    if 'pos_types' in graph.keys():
                        edge_index = adj_edge
                        edge_weight = torch.tensor(graph['given_types'], dtype=torch.float32)
                    else:
                        edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                        edge_weight = torch.ones(edge_index.size(1))

                    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

                full_edges = []

                for edge in given_edges:
                    full_edges.append(edge)

                for edge in train_pos:
                    full_edges.append(edge)

                full_adj_edge = torch.transpose(torch.tensor(full_edges), 1, 0)
                full_edge_index = torch.cat((full_adj_edge,  full_adj_edge[[1,0]]), dim=1)
                full_edge_weight = torch.ones(full_edge_index.size(1))
                full_adj = SparseTensor.from_edge_index(full_edge_index, full_edge_weight, [num_nodes, num_nodes])
                    
                graph_data = {}
                graph_data['adj'] = adj
                graph_data['full_adj'] = full_adj
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


def train(model, score_func, graph, optimizer, device, with_loss_weight):
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    full_adj = graph['full_adj'].to(device)

    optimizer.zero_grad()
    h = model(x, adj)

    inner_prod = torch.sigmoid(torch.mm(h, h.t()))

    ###############
    if with_loss_weight:
        # print('using loss weight')
        pos_weight = float(full_adj.size(0) * full_adj.size(0) - full_adj.sum()) / full_adj.sum()
        weight_mask = full_adj.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(x.device)
        weight_tensor[weight_mask] = pos_weight
    #########################

        loss = F.binary_cross_entropy(inner_prod.view(-1), full_adj.to_dense().view(-1), weight = weight_tensor)
    else:
        loss = F.binary_cross_entropy(inner_prod.view(-1), full_adj.to_dense().view(-1))

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_edge(score_func, graph, edges, model, device):
    pos_preds = []
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    h = model(x, adj)
    edges = edges
    pos_scores = score_func(h[edges[0]], h[edges[1]]).cpu()
    pos_preds += [pos_scores]
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds


@torch.no_grad()
def test(model, score_func, data, evaluator_hit, evaluator_mrr, device):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred = test_edge(score_func, graph, graph['pos'], model, device)
        neg_pred = test_edge(score_func, graph, graph['neg'], model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred = test_edge(score_func, graph, graph['pos'], model, device)
        neg_pred = test_edge(score_func, graph, graph['neg'], model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred = test_edge(score_func, graph, graph['pos'], model, device)
        neg_pred = test_edge(score_func, graph, graph['neg'], model, device)
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
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--input_size', type=int, default=602)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')

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
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    #####
    parser.add_argument('--with_loss_weight', default=False, action='store_true')


    args = parser.parse_args()
   
   
    # print(args.lr, args.l2, args.dropout)
    print(args.with_loss_weight)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind)

    input_channel = args.input_size
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)

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
        save_path = args.output_dir+'/best_run_'+str(run)

        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            score_func.train()
            loss = 0.0
            loss_count = 0
            
            start_time = time.time()
            for graph in data['train']:
                loss += train(model, score_func, graph, optimizer, device, args.with_loss_weight)
                loss_count +=1
                train_memory.append(torch.cuda.max_memory_allocated(device=None))
            train_times.append(time.time() - start_time)
        
            if epoch % args.eval_steps == 0:
                model.eval()
                score_func.eval()
                start_time = time.time()
                results_rank, score_emb = test(model, score_func, data, evaluator_hit, evaluator_mrr, device)
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
                              f'Loss: {loss / loss_count:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:
                       
                        save_model(model, save_path, emb=None)
                
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


    
    print(best_metric_valid_str)
    print("Training max memory (bytes):", max(train_memory))
    print("Testing max memory (bytes):", max(test_memory))
    print("Average total train time (s)", sum(train_times) / float(args.runs))
    print("Average total test time (s)", sum(test_times) / float(args.runs))
    print("Training run time per epoch (s)", statistics.mean(train_times), "+-", statistics.stdev(train_times))
    print("Testing run times per epoch (s)", statistics.mean(test_times), "+-", statistics.stdev(test_times))

    return best_valid_mean_metric, result_all_run
    
    



if __name__ == "__main__":
    main()
   