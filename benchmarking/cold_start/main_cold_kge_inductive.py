
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from red_gnn_model import *
from adaprop_model import *
from run_gnn_model import *
from utils import *
from scoring import mlp_score
import random
import time
import statistics
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

def load_graph(triples, n_ent, n_rel):
        
    KG = np.array(triples)
    idd = np.concatenate([np.expand_dims(np.arange(n_ent),1), 2*n_rel*np.ones((n_ent, 1)), np.expand_dims(np.arange(n_ent),1)], 1)
    KG = np.concatenate([KG, idd], 0)

    n_fact = KG.shape[0]
    M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:,0])), shape=(n_fact, n_ent))
    
    return KG, M_sub

log_print		= get_logger('testrun', 'log', get_config_dir())
def read_data(data_name, dir_path, cold_perc, blind, n_rel):
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
                if (len(given_edges) > 0):
                    given_head = []
                    given_tail = []
                    pos_head = []
                    pos_tail = []
                    neg_head = []
                    neg_tail = []
                    given_triples = []
                    pos_triples = []
                    neg_triples = []

                    '''for edge in given_edges:
                        given_head.append(edge[0])
                        given_tail.append(edge[1])
                        given_head.append(edge[1])
                        given_tail.append(edge[0])

                    for edge in train_pos:
                        pos_head.append(edge[0])
                        pos_tail.append(edge[1])
                        pos_head.append(edge[1])
                        pos_tail.append(edge[0])

                    for edge in train_neg:
                        neg_head.append(edge[0])
                        neg_tail.append(edge[1])
                        neg_head.append(edge[1])
                        neg_tail.append(edge[0])'''

                    if 'given_types' in graph.keys():
                        for i in range(len(given_edges)):
                            given_triples.append([given_edges[i][0], graph['given_types'][i], given_edges[i][1]])
                            given_triples.append([given_edges[i][1], graph['given_types'][i] + n_rel, given_edges[i][0]])
                    else:
                        for i in range(len(given_edges)):
                            given_triples.append([given_edges[i][0], 0, given_edges[i][1]])
                            given_triples.append([given_edges[i][1], 1, given_edges[i][0]])

                    if 'pos_types' in graph.keys():
                        for i in range(len(train_pos)):
                            pos_triples.append([train_pos[i][0], graph['pos_types'][i], train_pos[i][1]])
                            pos_triples.append([train_pos[i][1], graph['pos_types'][i] + n_rel, train_pos[i][0]])
                    else:
                        for i in range(len(train_pos)):
                            pos_triples.append([train_pos[i][0], 0, train_pos[i][1]])
                            pos_triples.append([train_pos[i][1], 1, train_pos[i][0]])

                    if 'neg_types' in graph.keys():
                        for i in range(len(train_neg)):
                            neg_triples.append([train_neg[i][0], graph['neg_types'][i], train_neg[i][1]])
                            neg_triples.append([train_neg[i][1], graph['neg_types'][i] + n_rel, train_neg[i][0]])
                    else:
                        for i in range(len(train_neg)):
                            neg_triples.append([train_neg[i][0], 0, train_neg[i][1]])
                            neg_triples.append([train_neg[i][1], 1, train_neg[i][0]])
                    
                    graph_data = {}
                    graph_data['given_head'] = torch.tensor(given_triples, dtype=torch.long)[:,0]
                    graph_data['given_relation'] = torch.tensor(given_triples, dtype=torch.long)[:,1]
                    graph_data['given_tail'] = torch.tensor(given_triples, dtype=torch.long)[:,2]
                    graph_data['pos_head'] = torch.tensor(pos_triples, dtype=torch.long)[:,0]
                    graph_data['pos_relation'] = torch.tensor(pos_triples, dtype=torch.long)[:,1]
                    graph_data['pos_tail'] = torch.tensor(pos_triples, dtype=torch.long)[:,2]
                    graph_data['neg_head'] = torch.tensor(neg_triples, dtype=torch.long)[:,0]
                    graph_data['neg_relation'] = torch.tensor(neg_triples, dtype=torch.long)[:,1]
                    graph_data['neg_tail'] = torch.tensor(neg_triples, dtype=torch.long)[:,2]
                    graph_data['x'] = graph['gnn_feature']
                    graph_data['node_count'] = num_nodes
                    graph_data['given_graph'], graph_data['given_subs'] = load_graph(given_triples, num_nodes, n_rel)

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
        

def train(model, graph, optimizer, device):
    given_head = graph['given_head']
    given_relation = graph['given_relation']
    head = graph['pos_head']
    relation = graph['pos_relation']
    tail = graph['pos_tail']
    given_graph = graph['given_graph']
    given_subs = graph['given_subs']

    optimizer.zero_grad()
    scores = model(head, relation, graph['node_count'], given_graph, given_subs, device)
    pos_scores = scores[[torch.arange(len(scores)),tail.to(device)]]
    max_n = torch.max(scores, 1, keepdim=True)[0]
    loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # avoid NaN
    for p in model.parameters():
        X = p.data.clone()
        flag = X != X
        X[flag] = np.random.random()
        p.data.copy_(X)

    return loss.item()



@torch.no_grad()
def test_edge(graph, edges, model, device):
    pos_preds = []
    given_graph = graph['given_graph']
    given_subs = graph['given_subs']

    if edges == 'pos':
        head = graph['pos_head']
        relation = graph['pos_relation']
        tail = graph['pos_tail']
    else:
        head = graph['pos_head']
        relation = graph['pos_relation']
        tail = graph['pos_tail']

    scores = model(head, relation, graph['node_count'], given_graph, given_subs, device)
    pos_scores = scores[[torch.arange(len(scores)).cpu(),tail.cpu()]]
    pos_preds += [pos_scores]
          
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds


@torch.no_grad()
def test(model, data, evaluator_hit, evaluator_mrr, device):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred = test_edge(graph, 'pos', model, device)
        neg_pred = test_edge(graph, 'neg', model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred = test_edge(graph, 'pos', model, device)
        neg_pred = test_edge(graph, 'neg', model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred = test_edge(graph, 'pos', model, device)
        neg_pred = test_edge(graph, 'neg', model, device)
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
    parser.add_argument('--model', type=str, default='RED_GNN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')
    parser.add_argument('--max_nodes', type=int, default=50)
    parser.add_argument('--num_relations', type=int, default=1)
    parser.add_argument('--attn_dim', type=int, default=5)

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

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind, args.num_relations)

    input_channel = args.input_size

    model = eval(args.model)(args.num_layers, args.hidden_channels, args.attn_dim, args.num_relations, args.dropout).to(device)
    
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

        #model.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0

        for epoch in range(1, 1 + args.epochs):
            model.train()
            loss = 0.0
            loss_count = 0
            
            start_time = time.time()
            for graph in data['train']:
                loss += train(model, graph, optimizer, device)
                loss_count +=1
                train_memory.append(torch.cuda.max_memory_allocated(device=None))
            train_times.append(time.time() - start_time)
            
            if epoch % args.eval_steps == 0:
                model.eval()
                start_time = time.time()
                results_rank, score_emb = test(model, data, evaluator_hit, evaluator_mrr, device)
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


    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   