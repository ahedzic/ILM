
import sys
sys.path.append("..") 

import torch
import numpy as np
import argparse
import scipy.sparse as ssp
import pickle
import time
import statistics
from gnn_model import *
from utils import *
from scoring import mlp_score
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

from baseline_models.neognn import NeoGNN
from torch_scatter import scatter_add

dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

def read_data(data_name, dir_path, beta, cold_perc, blind):
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

            if (len(train_pos) > 0) and (len(train_neg) > 0):
                given_edges = graph['given_edges']
                num_nodes = graph['node_count']
                positive_edges = train_pos
                negative_edges = train_neg

                if (cold_perc == 0.0) or (len(given_edges) == 0):
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), sparse_sizes=[num_nodes, num_nodes])
                    A = ssp.csr_matrix(torch.zeros((num_nodes, num_nodes)), shape=(num_nodes, num_nodes))
                else:
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)

                    if 'pos_types' in graph.keys() and len(graph['pos_types']):
                        edge_index = adj_edge
                        edge_weight = torch.tensor(graph['given_types'], dtype=torch.float32)
                    else:
                        edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                        edge_weight = torch.ones(edge_index.size(1))

                    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

                    edge_index = edge_index.cpu()
                    #edge_weight = torch.ones(edge_index.size(1), dtype=float)
                    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                                        shape=(num_nodes, num_nodes))
                    
                # Add loop nodes to themselves otherwise the model fails on isolated nodes
                for i in range(num_nodes):
                    A[i, i] = 1.0

                A = A.astype(np.double)
                A2 = A * A
                A = A + beta*A2
                        
                graph_data = {}
                graph_data['adj'] = adj
                graph_data['A'] = A
                graph_data['pos'] = torch.tensor(positive_edges)
                graph_data['neg'] = torch.tensor(negative_edges)
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

        

def train(model, score_func, graph, optimizer, args):
    train_pos = graph['pos']
    train_neg = graph['neg']
    x = graph['x']
    A = graph['A']
    adj = graph['adj']

    optimizer.zero_grad()


    num_nodes = x.size(0)

    # print(adj)

    edge = train_pos.t()
    pos_out, pos_out_struct, pos_out_feat, _ = model(edge,  adj, A, x, num_nodes, score_func, only_feature=args.only_use_feature)
        
    edge = train_neg.t()
    if len(edge) > 0:
        neg_out, neg_out_struct, neg_out_feat, _ = model(edge, adj, A, x, num_nodes, score_func, only_feature=args.only_use_feature)



    if pos_out_struct != None:
        pos_loss = -torch.log(pos_out_struct + 1e-15).mean()

        if len(edge) > 0:
            neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
        else:
            neg_loss = 0

        loss1 = pos_loss + neg_loss
    else:
        loss1 = 0


    if pos_out_feat != None:
        pos_loss = -torch.log(pos_out_feat + 1e-15).mean()

        if len(edge) > 0:
            neg_loss = -torch.log(1 - neg_out_feat + 1e-15).mean()
        else:
            neg_loss = 0
        
        loss2 = pos_loss + neg_loss
    else:
        loss2 = 0

    if pos_out != None:
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        if len(edge) > 0:
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        else:
            neg_loss = 0
        
        loss3 = pos_loss + neg_loss
    else:
        loss3 = 0

    loss = loss1 + loss2 + loss3
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
    optimizer.step()

    return loss.item()



@torch.no_grad()
def test_edge(score_func, graph, edges, model):
    h = model.forward_feature(graph['x'], graph['adj'])
    A = graph['A']
    num_nodes = graph['node_count']

    edge_weight = torch.from_numpy(A.data).to(h.device)
    edge_weight = model.f_edge(edge_weight.unsqueeze(-1))

    row, col = A.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(h.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg =  model.f_node(deg).squeeze()

    deg = deg.cpu().numpy()
    A_ = A.multiply(deg).tocsr()

    alpha = torch.softmax(model.alpha, dim=0).cpu()
  
    preds = []
    t_edges = edges.t()
    gnn_scores = score_func(h[t_edges[0]], h[t_edges[1]]).squeeze().cpu()
    src, dst = t_edges.cpu()
    cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
    cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())  
    cur_scores = alpha[0]*cur_scores + alpha[1] * gnn_scores
    preds += [cur_scores]

    if len(cur_scores.size()) > 0:
        pred_all = torch.cat(preds, dim=0)
    else:
        pred_all = torch.tensor(preds)

    return pred_all


@torch.no_grad()
def test(model, score_func, data, evaluator_hit, evaluator_mrr):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred = test_edge(score_func, graph, graph['pos'], model)
        neg_pred = test_edge(score_func, graph, graph['neg'], model)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred = test_edge(score_func, graph, graph['pos'], model)
        neg_pred = test_edge(score_func, graph, graph['neg'], model)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred = test_edge(score_func, graph, graph['pos'], model)
        neg_pred = test_edge(score_func, graph, graph['neg'], model)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        test_results.append((hits, mrr))
    
    result = get_average_results(train_results, valid_results, test_results)
    
    score_emb = [pos_pred.cpu(),neg_pred.cpu(), pos_pred.cpu(), neg_pred.cpu()]

    return result, score_emb

def data_to_device(data_set, device):

    for data in data_set['train']:
        data['adj'] = data['adj'].to(device)
        #data['A'] = data['A'].to(device)
        data['pos'] = data['pos'].to(device)
        data['neg'] = data['neg'].to(device)
        data['x'] = data['x'].to(device)

    for data in data_set['train_valid']:
        data['adj'] = data['adj'].to(device)
        #data['A'] = data['A'].to(device)
        data['pos'] = data['pos'].to(device)
        data['neg'] = data['neg'].to(device)
        data['x'] = data['x'].to(device)

    for data in data_set['valid']:
        data['adj'] = data['adj'].to(device)
        #data['A'] = data['A'].to(device)
        data['pos'] = data['pos'].to(device)
        data['neg'] = data['neg'].to(device)
        data['x'] = data['x'].to(device)

    for data in data_set['test']:
        data['adj'] = data['adj'].to(device)
        #data['A'] = data['A'].to(device)
        data['pos'] = data['pos'].to(device)
        data['neg'] = data['neg'].to(device)
        data['x'] = data['x'].to(device)

    return data_set

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='NeoGNN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--input_size', type=int, default=602)
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
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
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

    ######neo-gnn
    parser.add_argument('--f_edge_dim', type=int, default=8) 
    parser.add_argument('--f_node_dim', type=int, default=128) 
    parser.add_argument('--g_phi_dim', type=int, default=128) 
    parser.add_argument('--only_use_feature',	action='store_true',   default=False,   	help='whether only use the feature')
    parser.add_argument('--beta', type=float, default=0.1)
	



    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
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

    data = read_data(args.data_name, args.input_dir, args.beta, args.cold_perc, args.blind)

    input_channel = data['train'][0]['x'].size(1)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout,args).to(device)
       
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    data = data_to_device(data, device)

    # x = data['x'].to(device)
    # train_pos = data['train_pos'].to(x.device)

    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
       
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

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        save_path = args.output_dir+'/best_run_'+str(run)


        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            score_func.train()
            loss = 0.0
            loss_count = 0

            start_time = time.time()
            for graph in data['train']:
                loss += train(model, score_func, graph, optimizer, args)
                loss_count +=1
                train_memory.append(torch.cuda.max_memory_allocated(device=None))
            train_times.append(time.time() - start_time)
            # print(model.convs[0].att_src[0][0][:10])
           
            if epoch % args.eval_steps == 0:
                model.eval()
                score_func.eval()
                start_time = time.time()
                results_rank, emb = test(model, score_func, data, evaluator_hit, evaluator_mrr)
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
   