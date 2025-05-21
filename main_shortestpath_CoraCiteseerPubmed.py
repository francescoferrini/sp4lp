import argparse
import time
import sys
import os

current_dir = os.path.dirname(__file__)
utils_path = os.path.abspath(os.path.join(current_dir, '..', 'utils'))
sys.path.append(utils_path)

import itertools
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model_small import *
from utils import *
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
torch.set_num_threads(4) 
import networkx as nx

from ogb.linkproppred import Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
import pickle


log_print		= get_logger('testrun', 'log', get_config_dir())
def read_data(data_name, dir_path, filename):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:

        
        path = dir_path+ '/{}/{}_pos.txt'.format(data_name, split)
      
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f'{dir_path}/{data_name}/heart_valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)


    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    
    test_pos =  torch.tensor(test_pos)
    

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]


    feature_embeddings = torch.load(dir_path+ '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings

    return data
    


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

   
    result = {}
    k_list = [1, 3, 10, 100]
   
    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in [1,3,10, 100]:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    
    return result

        
def train_path_batch(model, optimizer, train_pos, x, batch_size):
    model.train()
    total_loss, total_examples = 0, 0
    num_nodes = x.size(0)

    # Grafo iniziale completo
    full_edge_list = train_pos.tolist()
   
    G_full = nx.Graph()
    G_full.add_edges_from(full_edge_list)
    G_full.add_nodes_from(range(num_nodes))

    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        masked_edges = train_pos[mask].t()
        masked_edges = torch.cat([masked_edges, masked_edges[[1, 0]]], dim=1)

        edge_weight = torch.ones(masked_edges.size(1)).to(torch.float).to(train_pos.device)
        adj = SparseTensor.from_edge_index(masked_edges, edge_weight, [num_nodes, num_nodes]).to(train_pos.device)

        # GNN embeddings
        edge_index = masked_edges
        h = model.encoder(x, edge_index)

        batch_edges = train_pos[perm]
        u, v = batch_edges[:, 0], batch_edges[:, 1]

        # Path computation
        G_batch = G_full.copy()
        G_batch.remove_edges_from(batch_edges.tolist())

        batch_paths = []
        for u_i, v_i in zip(u.tolist(), v.tolist()):
            try:
                path = nx.shortest_path(G_batch, u_i, v_i)
            except:
                path = [u_i, v_i]
            batch_paths.append(torch.tensor(path, dtype=torch.long, device=h.device))
        
        # Forward full model
        preds = model(x, edge_index, batch_edges.to(h.device), batch_paths)

        # Positive loss
        pos_loss = -torch.log(preds + 1e-15).mean()

        # Negative sampling
        neg_edges = torch.randint(0, num_nodes, batch_edges.shape, device=h.device)

        u_, v_ = neg_edges[:, 0], neg_edges[:, 1]
        fake_paths = []
        for u_i, v_i in zip(u_.tolist(), v_.tolist()):
            try:
                path = nx.shortest_path(G_batch, u_i, v_i)
            except:
                path = [u_i, v_i]
            fake_paths.append(torch.tensor(path, dtype=torch.long, device=h.device))
    
        neg_preds = model(x, edge_index, neg_edges.to(h.device), fake_paths)
        neg_loss = -torch.log(1 - neg_preds + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_edges.size(0)
        total_examples += batch_edges.size(0)

    return total_loss / total_examples

@torch.no_grad()
def test_edge_with_path(model, input_data, x, full_edge_index, batch_size, negative_data=None):
    pos_preds = []
    neg_preds = []

    import networkx as nx
    num_nodes = x.size(0)

    edge_list = full_edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    G.add_nodes_from(range(num_nodes))

    # h = model.encoder(x, full_edge_index)

    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm]
            neg_edges = negative_data[perm]  # [B, 500, 2]

            # Positive path
            pos_paths = []
            for u, v in pos_edges.tolist():
                try:
                    path = nx.shortest_path(G, u, v)
                except:
                    path = [u, v]
                pos_paths.append(torch.tensor(path, dtype=torch.long, device=x.device))

            pos_scores = model(x, full_edge_index, pos_edges.to(x.device), pos_paths).cpu()
            pos_preds.append(pos_scores)

            # Negative paths (500 per esempio)
            batch_neg = []
            for paths500 in neg_edges:  # shape: [500, 2]
                temp_paths = []
                for u, v in paths500.tolist():
                    try:
                        path = nx.shortest_path(G, u, v)
                    except:
                        path = [u, v]
                    temp_paths.append(torch.tensor(path, dtype=torch.long, device=x.device))
                edges500 = paths500.to(x.device)
                scores500 = model(x, full_edge_index, edges500, temp_paths).cpu()  # [500]
                batch_neg.append(scores500.unsqueeze(0))  # [1, 500]

            batch_neg = torch.cat(batch_neg, dim=0)  # [B, 500]
            neg_preds.append(batch_neg)

        neg_preds = torch.cat(neg_preds, dim=0)  # [N, 500]

    else:
        neg_preds = None
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm]
            pos_paths = []
            for u, v in pos_edges.tolist():
                try:
                    path = nx.shortest_path(G, u, v)
                except:
                    path = [u, v]
                pos_paths.append(torch.tensor(path, dtype=torch.long, device=x.device))

            pos_scores = model(x, full_edge_index, pos_edges.to(x.device), pos_paths).cpu()
            pos_preds.append(pos_scores)

    pos_preds = torch.cat(pos_preds, dim=0)
    return pos_preds, neg_preds


@torch.no_grad()
def test_with_path(model, data, x, evaluator_hit, evaluator_mrr, batch_size, device):
    model.eval()
    x = x.to(device)
    full_edge_index = data['adj'].coo()[:2]
    full_edge_index = torch.stack(full_edge_index, dim=0).to(x.device)

    pos_train_pred, _ = test_edge_with_path(model, data['train_val'], x, full_edge_index, batch_size)
    pos_valid_pred, neg_valid_pred = test_edge_with_path(model, data['valid_pos'], x, full_edge_index, batch_size, negative_data=data['valid_neg'])
    pos_test_pred, neg_test_pred = test_edge_with_path(model, data['test_pos'], x, full_edge_index, batch_size, negative_data=data['test_neg'])


    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())

    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)


    return result


def train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()

    total_loss = total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()


        num_nodes = x.size(0)

        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
    
        train_edge_mask = train_pos[mask].transpose(1,0)

        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

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
def test_edge(score_func, input_data, h, batch_size, negative_data=None):

    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds


@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()
    
    h = model(x, data['adj'].to(x.device))
    x = h


    pos_train_pred, _ = test_edge(score_func, data['train_val'], h, batch_size)
    pos_valid_pred, neg_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size, negative_data=data['valid_neg'])
    pos_test_pred, neg_test_pred = test_edge(score_func, data['test_pos'], h, batch_size, negative_data=data['test_neg'])


    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result, score_emb

from tqdm import tqdm
def compute_shortest_paths(G, edge_pairs):
    paths = []
    for u, v in tqdm(edge_pairs, desc="Calculate shortest paths"):
        try:
            path = nx.shortest_path(G, source=u, target=v)
        except nx.NetworkXNoPath:
            path = [u, v]  
        paths.append(path)
    return paths



def generate_configs(grid_dict):
    keys = list(grid_dict.keys())
    values = list(itertools.product(*[grid_dict[k] for k in keys]))
    return [dict(zip(keys, v)) for v in values]

def run_all_configs(x, train_pos, data, args, device):
    grid_config = {
    'gnn_type': ['gcn'],
    'gnn_layers': [1],
    'hidden_dim': [256],
    'dropout': [0.2],
    'path_encoder_type': ['transformer'],
    'path_encoder_layers': [3],
    'path_encoder_heads': [4],
    'path_hidden_dim': [256],
    'predictor_hidden': [512],
    'input_dim': [x.size(1)]  
    }
    configs = generate_configs(grid_config)
    all_results = []

    for idx, config in enumerate(configs):
        print(f"\n========== Running config {idx + 1}/{len(configs)} ==========")
        print(config)

        model = build_model_from_config(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_valid, best_auc, result_dict = main_training_loop(
            model, optimizer, x, train_pos, data, args, device
        )

        all_results.append((config, best_valid, best_auc, result_dict))

    return all_results


def main_training_loop(model, optimizer, x, train_pos, data, args, device):
        for run in range(args.runs):
            print('#################################          ', run, '          #################################')
            if args.runs == 1:
                seed = args.seed
            else:
                seed = run
            print('seed: ', seed)
            init_seed(seed)
            eval_metric = args.metric
            evaluator_hit = Evaluator(name='ogbl-collab')
            evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

            loggers = {
                'Hits@1': Logger(args.runs),
                'Hits@3': Logger(args.runs),
                'Hits@10': Logger(args.runs),
                'Hits@100': Logger(args.runs),
                'MRR': Logger(args.runs),
            
            }
            optimizer = torch.optim.Adam(
                    list(model.parameters()),lr=args.lr, weight_decay=args.l2)
            
            best_valid = 0
            kill_cnt = 0
            for epoch in range(1, 1 + args.epochs):
                loss = train_path_batch(model, optimizer, train_pos, x, args.batch_size)
                
                if epoch % args.eval_steps == 0:
                    results_rank = test_with_path(model, data, x, evaluator_hit, evaluator_mrr, args.batch_size, device)
                    # results_rank, score_emb = test(model, score_func, data, x, evaluator_hit, evaluator_mrr, args.batch_size)

                    for key, result in results_rank.items():
                        loggers[key].add_result(run, result)

                    if epoch % args.log_steps == 0:
                        for key, result in results_rank.items():
                            
                            print(key)
                            
                            train_hits, valid_hits, test_hits = result
                        

                            log_print.info(
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
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

            print(best_metric_valid_str)
            best_auc_metric = best_valid_mean_metric
        return best_valid_mean_metric, best_auc_metric, result_all_run
    
    
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GIN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=1)
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

    data = read_data(args.data_name, args.input_dir, args.filename)

    x = data['x']

    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
        x = torch.cat((x, n2v_emb), dim=-1)

    x = x.to(device)
    train_pos = data['train_pos'].to(x.device)

    input_channel = x.size(1)
    
    all_results = run_all_configs(x, train_pos, data, args, device)
    print(all_results)
    with open("grid_search_results.pkl", "wb") as f:
        pickle.dump(all_results, f)



if __name__ == "__main__":
    main()
   