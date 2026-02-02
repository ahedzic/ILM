
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from relational_gnn_model import *
from utils import *
from meta_relational_scoring import meta_score
import random
import networkx as nx
import math
import time
import statistics
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import assortativity, to_networkx
from torch_geometric.data import Data


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

clust = []
isolated = []
mags = []

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
    
    for graphs_key in graphs.keys():
        for graph in graphs[graphs_key]:
            train_pos = graph['pos_edges']
            train_neg = graph['neg_edges']
            given_edges = graph['given_edges']
            num_nodes = graph['node_count']
            positive_edges = train_pos
            negative_edges = train_neg

            if (len(train_pos) > 0) and (len(train_neg) > 0):
                graph_data = {}

                if ('pos_types' in graph.keys()) and len(graph['pos_types']):
                    edge_types = torch.tensor(graph['pos_types'], dtype=torch.float)
                else:
                    edge_types = torch.zeros((len(train_pos), 1), dtype=torch.float)

                if (cold_perc == 0.0) or (len(given_edges) == 0):
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), value=torch.empty(0, dtype=torch.float32), sparse_sizes=[num_nodes, num_nodes])
                    data_adj = torch.empty(0, dtype=torch.long)
                    ntx_adj = nx.empty_graph(num_nodes)
                    graph_data['given_edges'] = []
                    edge_weight = torch.empty(0, dtype=torch.long)
                else:
                    graph_data['given_edges'] = torch.transpose(torch.tensor(given_edges), 1, 0)
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)
                    edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)

                    if len(graph['pos_types']):
                        if len(graph['given_types']):
                            edge_weight = torch.tensor(graph['given_types'], dtype=torch.long)
                            edge_weight = torch.cat((edge_weight,  edge_weight))
                        else:
                            edge_weight = torch.empty(0, dtype=torch.long)
                            
                        adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight, sparse_sizes=[num_nodes, num_nodes])
                    else:
                        edge_weight = torch.zeros(edge_index.size(1), dtype=torch.long)
                        adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

                    data_adj = edge_index
                    ntx_adj = to_networkx(Data(graph['gnn_feature'], data_adj))
                    
                graph_data['adj'] = adj
                graph_data['empty_adj'] = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), value=torch.empty(0, dtype=torch.float32), sparse_sizes=[num_nodes, num_nodes])
                graph_data['adj_type'] = edge_weight
                graph_data['data_adj'] = data_adj
                graph_data['ntx_adj'] = ntx_adj
                graph_data['pos'] = torch.transpose(torch.tensor(positive_edges), 1, 0)
                graph_data['pos_types'] = edge_types
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
        

def train(model, score_func, graph, optimizer, device, iterations):
    expected_pos = graph['pos'].to(device)
    expected_neg = graph['neg'].to(device)
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    empty_adj = graph['empty_adj'].to(device)
    edge_types = graph['adj_type'].to(device)
    ntx_adj = graph['ntx_adj']
    data_adj = graph['data_adj'].to(device)
    total_loss = total_examples = 0
    current_types = edge_types

    given_edges = graph['given_edges']
    num_nodes = graph['node_count']
    clustering_coeffs = torch.zeros((graph['node_count'], 1)).to(device)

    for _ in range(iterations):
        optimizer.zero_grad()
    
        # Calculate graph clustering coefficient
        coeffs = nx.clustering(ntx_adj)

        for key in coeffs.keys():
            clustering_coeffs[key] = coeffs[key]

        h = model(x, adj, current_types)
        h_empty = model(x, empty_adj, torch.empty(0, dtype=torch.long).to(device))
        type_loss = torch.nn.CrossEntropyLoss()
        edge = expected_pos
        pos_out, pos_add_score, pos_types, pos_e_types = score_func(x, h, h_empty, edge[0], edge[1], clustering_coeffs)
        edge = expected_neg
        neg_out, neg_add_score, neg_types, neg_e_types = score_func(x, h, h_empty, edge[0], edge[1], clustering_coeffs)

        if ('pos_types' in graph.keys()) and len(graph['pos_types']):
            pos_loss = -torch.log(pos_out + 1e-15).mean() + type_loss(pos_e_types, graph['pos_types'].to(device))
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean() + type_loss(neg_e_types, torch.zeros(neg_e_types.shape).to(device))
        else:
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        if _ != (iterations - 1):
            i_edge = []
            j_edge = []
            added_types = []

            for i in range(len(pos_out)):
                if pos_out[i] >= 0.5:
                    i_edge.append(expected_pos[0][i])
                    j_edge.append(expected_pos[1][i])
                    added_types.append(pos_types[i])

            for i in range(len(neg_out)):
                if neg_out[i] >= 0.5:
                    i_edge.append(expected_neg[0][i])
                    j_edge.append(expected_neg[1][i])
                    added_types.append(neg_types[i])

            new_edges = torch.stack([torch.tensor(i_edge), torch.tensor(j_edge)]).to(device)
            new_edges_mask = torch.cat((new_edges, new_edges[[1,0]]), dim=1).to(torch.long).to(device)
            new_types = torch.tensor(added_types).to(torch.int64).to(device)
            new_types = torch.cat([new_types, new_types]).to(torch.int64).to(device)

            if len(given_edges):
                train_edge_mask = torch.cat((given_edges, given_edges[[1,0]]), dim=1).to(device)
                new_total_edges = torch.cat([train_edge_mask, new_edges_mask], 1).to(torch.long).to(device)
                #new_edge_weight_mask = torch.ones(train_edge_mask.size(1) + new_edges_mask.size(1)).to(torch.float).to(device)
                new_edge_weight_mask = torch.cat([edge_types, new_types]).to(torch.int64).to(device)
                
                if ('pos_types' in graph.keys()) and len(graph['pos_types']):
                    adj = SparseTensor.from_edge_index(new_total_edges, edge_attr=new_edge_weight_mask, sparse_sizes=[num_nodes, num_nodes]).to(device)
                else:
                    adj = SparseTensor.from_edge_index(new_total_edges, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)

                data_adj = new_total_edges
                ntx_adj = to_networkx(Data(x, new_total_edges))
                current_types = new_edge_weight_mask
            else:
                new_edge_weight_mask = new_types
                
                if ('pos_types' in graph.keys()) and len(graph['pos_types']):
                    adj = SparseTensor.from_edge_index(new_edges_mask, edge_attr=new_edge_weight_mask, sparse_sizes=[num_nodes, num_nodes]).to(device)
                else:
                    adj = SparseTensor.from_edge_index(new_edges_mask, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)

                data_adj = new_edges_mask
                ntx_adj = to_networkx(Data(x, new_edges_mask))
                current_types = new_edge_weight_mask

            if (len(given_edges) == 0) and (len(i_edge) == 0):
                data_adj = torch.tensor([], dtype=torch.int64).to(device)

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)

        total_loss += loss.item() * num_examples
        total_examples += num_examples


    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, graph, pos_edges, neg_edges, model, device, iterations):
    pos_preds = []
    neg_preds = []
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    empty_adj = graph['empty_adj'].to(device)
    edge_types = graph['adj_type'].to(device)
    ntx_adj = graph['ntx_adj']
    data_adj = graph['data_adj'].to(device)
    given_edges = graph['given_edges']
    num_nodes = graph['node_count']
    clustering_coeffs = torch.zeros((graph['node_count'], 1)).to(device)
    current_types = edge_types

    for _ in range(iterations):
        # Calculate graph clustering coefficient
        coeffs = nx.clustering(ntx_adj)

        for key in coeffs.keys():
            clustering_coeffs[key] = coeffs[key]

        h = model(x, adj, current_types)
        h_empty = model(x, empty_adj, torch.empty(0, dtype=torch.long).to(device))
        edges = pos_edges
        pos_scores, pos_add_score, pos_types, pos_e_types = score_func(x, h, h_empty, edges[0], edges[1], clustering_coeffs)
        pos_scores = pos_scores.cpu()

        edges = neg_edges
        neg_scores, neg_add_score, neg_types, neg_e_types = score_func(x, h, h_empty, edges[0], edges[1], clustering_coeffs)
        neg_scores = neg_scores.cpu()

        if _ != (iterations - 1):
            i_edge = []
            j_edge = []
            added_types = []

            for i in range(len(pos_scores)):
                if pos_scores[i] >= 0.5:
                    i_edge.append(pos_edges[0][i])
                    j_edge.append(pos_edges[1][i])
                    added_types.append(pos_types[i])

            for i in range(len(neg_scores)):
                if neg_scores[i] >= 0.5:
                    i_edge.append(neg_edges[0][i])
                    j_edge.append(neg_edges[1][i])
                    added_types.append(neg_types[i])

            new_edges = torch.stack([torch.tensor(i_edge), torch.tensor(j_edge)]).to(device)
            new_edges_mask = torch.cat((new_edges, new_edges[[1,0]]), dim=1).to(torch.long).to(device)
            new_types = torch.tensor(added_types).to(torch.int64).to(device)
            new_types = torch.cat([new_types, new_types]).to(torch.int64).to(device)

            if len(given_edges):
                train_edge_mask = torch.cat((given_edges, given_edges[[1,0]]), dim=1).to(device)
                new_total_edges = torch.cat([train_edge_mask, new_edges_mask], 1).to(torch.long).to(device)
                #new_edge_weight_mask = torch.ones(train_edge_mask.size(1) + new_edges_mask.size(1)).to(torch.float).to(device)
                new_edge_weight_mask = torch.cat([edge_types, new_types]).to(torch.int64).to(device)
                
                if ('pos_types' in graph.keys()) and len(graph['pos_types']):
                    adj = SparseTensor.from_edge_index(new_total_edges, edge_attr=new_edge_weight_mask, sparse_sizes=[num_nodes, num_nodes]).to(device)
                else:
                    adj = SparseTensor.from_edge_index(new_total_edges, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)

                data_adj = new_total_edges
                ntx_adj = to_networkx(Data(x, new_total_edges))
                current_types = new_edge_weight_mask
            else:
                new_edge_weight_mask = new_types
                
                if ('pos_types' in graph.keys()) and len(graph['pos_types']):
                    adj = SparseTensor.from_edge_index(new_edges_mask, edge_attr=new_edge_weight_mask, sparse_sizes=[num_nodes, num_nodes]).to(device)
                else:
                    adj = SparseTensor.from_edge_index(new_edges_mask, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)

                data_adj = new_edges_mask
                ntx_adj = to_networkx(Data(x, new_edges_mask))
                current_types = new_edge_weight_mask

            if (len(given_edges) == 0) and (len(i_edge) == 0):
                data_adj = torch.tensor([], dtype=torch.int64).to(device)
    
    pos_preds += [pos_scores]
    neg_preds += [neg_scores]
          
    pos_preds = torch.cat(pos_preds, dim=0)
    neg_preds = torch.cat(neg_preds, dim=0)


    return pos_preds, neg_preds


@torch.no_grad()
def test(model, score_func, data, evaluator_hit, evaluator_mrr, device, iterations):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred, neg_pred = test_edge(score_func, graph, graph['pos'], graph['neg'], model, device, iterations)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred, neg_pred = test_edge(score_func, graph, graph['pos'], graph['neg'], model, device, iterations)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    global clust
    global isolated
    global mags
    clust = []
    isolated = []
    mags = []
    score_func.reset_meta_distributions()

    for graph in data['test']:  
        pos_pred, neg_pred = test_edge(score_func, graph, graph['pos'], graph['neg'], model, device, iterations)
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

    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind, args.num_relations)

    input_channel = args.input_size
    model = eval(args.gnn_model)(input_channel, args.hidden_channels, args.num_relations,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, args.max_nodes, args.cat_node_feat_mf).to(device)
    score_func = eval(args.score_model)(args.input_size, args.hidden_channels, args.hidden_channels,
                    1, args.num_relations, args.num_layers_predictor, args.dropout, args.weights, device).to(device)
    
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
        skip_training = False

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        model_save_path = args.output_dir+'/model_data'+args.data_name+'_model'+args.gnn_model+'_lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)
        score_save_path = args.output_dir+'/score_data'+args.data_name+'_model'+args.gnn_model+'_lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        if os.path.exists(model_save_path) and os.path.exists(score_save_path):
            model.load_state_dict(torch.load(model_save_path))
            score_func.load_state_dict(torch.load(score_save_path))
            skip_training = True
        else:
            model.reset_parameters()
            score_func.reset_parameters()
            score_func.reset_meta_distributions()

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            score_func.train()
            loss = 0.0
            loss_count = 0
            
            if not skip_training:
                start_time = time.time()
                for graph in data['train']:
                    loss += train(model, score_func, graph, optimizer, device, args.iterations)
                    loss_count +=1
                    train_memory.append(torch.cuda.max_memory_allocated(device=None))
                train_times.append(time.time() - start_time)
            
            if epoch % args.eval_steps == 0:
                model.eval()
                score_func.eval()
                score_func.reset_meta_distributions()
                start_time = time.time()
                results_rank, score_emb = test(model, score_func, data, evaluator_hit, evaluator_mrr, device, args.iterations)
                test_memory.append(torch.cuda.max_memory_allocated(device=None))
                test_times.append(time.time() - start_time)
                distribution = score_func.get_meta_distributions()
                score_func.reset_meta_distributions

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                       

                        if skip_training:
                            log_print.info(
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                        else:
                            log_print.info(
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {(loss / loss_count):.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()
                print("Meta distributions global(0 hop):", distribution[0], "global(1 hop):", distribution[1], "global(2 hop):", distribution[2], "sim(0 hop):", distribution[3], "sim(1 hop):", distribution[4], "sim(2 hop):", distribution[5], "diff(0 hop):", distribution[6], "diff(1 hop):", distribution[7], "diff(2 hop):", distribution[8])

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:

                        save_emb(score_emb, save_path)

                
                else:
                    kill_cnt += 1
                    
                    if (kill_cnt > args.kill_cnt) or skip_training: 
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
   
