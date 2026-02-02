
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from leap_model_hetero import Model, connector_model
from MLP_hetero import *
from utils import *
import random
import networkx as nx
import math
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
                    edge_types = torch.tensor(graph['pos_types'], dtype=torch.long)
                else:
                    edge_types = torch.zeros((len(train_pos), 1), dtype=torch.float)

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
                    graph_data['given_edges'] = torch.transpose(torch.tensor(given_edges), 1, 0)
                else:
                    adj_edge = torch.cat((adj_pos_edge,  adj_pos_edge[[1,0]]), dim=1)

                    if 'pos_types' in graphs.keys() and len(graph['pos_types']):
                        pos_types = torch.tensor(graph['pos_types'], dtype=torch.long)
                        edge_weights = torch.cat((pos_types, pos_types))
                    else:
                        edge_weights = torch.ones(adj_edge.size(1))

                    adj_train_edge = adj_edge
                    train_edge_weights = edge_weights
                    graph_data['given_edges'] = torch.empty(0, dtype=torch.long)

                graph_data['edge_index'] = adj_edge
                graph_data['edge_weight'] = edge_weights
                graph_data['x'] = graph['gnn_feature']
                graph_data['pos'] = torch.transpose(torch.tensor(positive_edges), 1, 0)
                graph_data['neg'] = torch.transpose(torch.tensor(negative_edges), 1, 0)
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

def train(model, connector, graph, optimizer, target_perc, device):
    x_feats = graph['x'].to(device)
    edge_index = graph['edge_index'].to(device)
    total_loss = total_examples = 0

    optimizer.zero_grad()

    num_nodes = graph['node_count']
    cce = nn.CrossEntropyLoss()
    targets = int(num_nodes * target_perc)
    inputs = num_nodes - targets

    if targets > 1 and inputs > 1:
        mlp_samples = MLPSamples(edge_index, num_inputs=inputs, num_targets=targets, device=device)
        cc = mlp_samples.create_component()
        mlp_samples.sample() 

        if (mlp_samples.targets.shape[0] > 1) and (mlp_samples.inputs.shape[0] > 1) and (mlp_samples.target_edges != None):       
            msg_edge_index = torch.tensor([ (u, v) for u, v in edge_index.T  
                                        if u not in set(mlp_samples.inputs) and v not in set(mlp_samples.inputs)]).T
            target_weight = 1 / (mlp_samples.shortest_path_lengths)
            
            target_edge_index = mlp_samples.target_edges.T.to(device).long()
            src_x = x_feats[target_edge_index.T[:, 0]]
            trg_x = x_feats[target_edge_index.T[:, 1]]
            src_x = connector(src_x)
            trg_x = connector(trg_x)
            target_edge_weights = src_x.mul(trg_x).sum(dim=-1)

            x, pred = model(x = x_feats, 
                        message_edge_index = msg_edge_index.to(device), 
                        target_edge_index = target_edge_index,
                        target_edge_weights = target_edge_weights,
                        mlp_inputs=mlp_samples.inputs.to(device))
            
            loss = model.loss(edge_index, num_nodes, x)

            if len(target_edge_weights.flatten()) == len(target_weight.flatten()):
                loss +=  0.0005 * cce(target_edge_weights.flatten(), target_weight.flatten())
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_examples = pred.shape[0]

            total_loss += loss.item() * num_examples
            total_examples += num_examples

            return total_loss / total_examples
        else:
            return 0
    else:
        return 0

def create_test_target_edges(inputs, targets):
    target_edges = []
    for node in inputs:
        edges = zip([node] * targets.shape[0], targets)
        target_edges.append(torch.tensor(list(edges)))

    if len(target_edges):
        target_edges = torch.cat(target_edges) 
    else:
        target_edges = None 

    return target_edges

@torch.no_grad()
def test_edge(connector, graph, pos_edges, neg_edges, model, target_perc, device):
    pos_preds = []
    neg_preds = []
    x = graph['x'].to(device)
    given_edges = graph['given_edges'].to(device)
    num_nodes = graph['node_count']

    #All nodes are disconnected so select random anchor nodes since pagerank won't work
    inputs = range(num_nodes)
    targets = np.random.choice(inputs, size=(int(num_nodes * target_perc),), replace=False)

    target_edges = create_test_target_edges(inputs, targets)

    if len(given_edges) == 0:
        given_edges = None

    src_x = x[target_edges.T[:, 0]]
    trg_x = x[target_edges.T[:, 1]]
    src_x = connector(src_x)
    trg_x = connector(trg_x)
    edge_weights = src_x.mul(trg_x).sum(dim=-1)

    scores_x, out = model(x=x,
                    message_edge_index=given_edges,
                    target_edge_index=target_edges.to(device),
                    target_edge_weights=edge_weights.to(device),
                    mlp_inputs=None)
    
    s_emb = scores_x[pos_edges.T[:, 0]].detach().cpu()
    t_emb = scores_x[pos_edges.T[:, 1]].detach().cpu()
    pos_scores = s_emb.mul(t_emb).sum(dim=-1)

    s_emb = scores_x[neg_edges.T[:, 0]].detach().cpu()
    t_emb = scores_x[neg_edges.T[:, 1]].detach().cpu()
    neg_scores = s_emb.mul(t_emb).sum(dim=-1)
    
    pos_preds += [pos_scores]
    neg_preds += [neg_scores]
          
    pos_preds = torch.cat(pos_preds, dim=0)
    neg_preds = torch.cat(neg_preds, dim=0)


    return pos_preds, neg_preds


@torch.no_grad()
def test(model, score_func, data, evaluator_hit, evaluator_mrr, target_perc, device):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred, neg_pred = test_edge(score_func, graph, graph['pos'], graph['neg'], model, target_perc, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred, neg_pred = test_edge(score_func, graph, graph['pos'], graph['neg'], model, target_perc, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred, neg_pred = test_edge(score_func, graph, graph['pos'], graph['neg'], model, target_perc, device)
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
    parser.add_argument('--target_perc', type=float, default=0.1)

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind, args.num_relations)

    input_channel = args.input_size

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

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        connect_model = connector_model(args.input_size, args.hidden_channels).to(args.device)
        model = Model(input_dim=args.input_size, num_targets=args.hidden_channels, dropout=args.dropout, device=args.device).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            connect_model.train()
            loss = 0.0
            loss_count = 0
            
            for graph in data['train']:
                loss += train(model, connect_model, graph, optimizer, args.target_perc, device)
                loss_count +=1
            
            if epoch % args.eval_steps == 0:
                model.eval()
                connect_model.eval()
                results_rank, score_emb = test(model, connect_model, data, evaluator_hit, evaluator_mrr, args.target_perc, device)

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

    global clust
    global isolated
    global mags

    with open(args.data_name + 'output_mags.csv', 'w') as file:
        for row in mags:
            file.write(str(row) + ',\n')

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   
