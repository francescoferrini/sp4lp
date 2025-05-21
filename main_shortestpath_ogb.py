
import sys
sys.path.append("..") 

import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
import pickle

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr
# from evaluate_mrr_hit import evaluate_mrr
from torch_geometric.utils import negative_sampling
from torch.nn.utils.rnn import pad_sequence
import os
import networkx as nx

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

from tqdm import tqdm
import itertools


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result


def test_path_batch(model, pos_edges, neg_edges, data, emb, adj,
                    precomputed_pos_paths, precomputed_neg_paths, batch_size):
    model.eval()
    x = data.x if emb is None else emb.weight
    device = x.device

    pos_preds = []
    neg_preds = []

    for start_idx in tqdm(range(0, pos_edges.size(0), batch_size), desc="Evaluating Positives"):
        end_idx = min(start_idx + batch_size, pos_edges.size(0))
        pos_batch = pos_edges[start_idx:end_idx].to(device)  
        pos_paths = [torch.tensor(precomputed_pos_paths[tuple(e.tolist())], dtype=torch.long, device=device)
                     for e in pos_batch]
        pos_scores = model(x, adj, pos_batch, pos_paths)
        pos_preds.append(pos_scores)

    neg_edges_flat = neg_edges.view(-1, 2)

    for start_idx in tqdm(range(0, neg_edges_flat.size(0), batch_size), desc="Evaluating Negatives"):
        end_idx = min(start_idx + batch_size, neg_edges_flat.size(0))
        neg_batch = neg_edges_flat[start_idx:end_idx].to(device)
        neg_paths = [torch.tensor(precomputed_neg_paths[tuple(e.tolist())], dtype=torch.long, device=device)
                     for e in neg_batch]
        neg_scores = model(x, adj, neg_batch, neg_paths)
        neg_preds.append(neg_scores)

    pos_pred = torch.cat(pos_preds, dim=0) 
    neg_pred = torch.cat(neg_preds, dim=0).view(pos_edges.size(0), -1)  

    return pos_pred, neg_pred


def train_path_batch(model, edges_for_loss, neg_edges_for_loss, data, emb, optimizer,
                     batch_size, pos_train_weight,
                     precomputed_pos_paths, precomputed_neg_paths,
                     G_message_passing, adj, train_loader):

    model.train()
    total_loss = total_examples = 0
    x = data.x if emb is None else emb.weight
    device = x.device

    for step_idx, perm in enumerate(tqdm(train_loader, desc="Training Batches")):
        optimizer.zero_grad()
        pos_batch = [edges_for_loss[i] for i in perm]
        pos_batch_tensor = torch.tensor(pos_batch, dtype=torch.long, device=device)
        pos_paths = [torch.tensor(precomputed_pos_paths[tuple(e)], dtype=torch.long, device=device) for e in pos_batch]

        preds = model(x, adj, pos_batch_tensor, pos_paths)
        pos_loss = -torch.log(preds + 1e-15).mean()
        neg_batch = [neg_edges_for_loss[i] for i in perm]
        neg_batch_tensor = torch.tensor(neg_batch, dtype=torch.long, device=device)
        neg_paths = [torch.tensor(precomputed_neg_paths[tuple(e)], dtype=torch.long, device=device) for e in neg_batch]

        neg_preds = model(x, adj, neg_batch_tensor, neg_paths)
        neg_loss = -torch.log(1 - neg_preds + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()

        total_loss += loss.item() * len(pos_batch)
        total_examples += len(pos_batch)
    return total_loss / total_examples


def preprocess_shortest_paths(train_pos, num_nodes):
    import networkx as nx
    from tqdm import tqdm

    G = nx.Graph()
    G.add_edges_from(train_pos.tolist())
    G.add_nodes_from(range(num_nodes))
    path_dict = {}
    for u, v in tqdm(train_pos.tolist(), desc="Precomputing paths"):
        try:
            path = nx.shortest_path(G, u, v)
        except:
            path = [u, v]
        path_dict[(u, v)] = torch.tensor(path, dtype=torch.long)

    
    return path_dict

def preprocess_path_tensor(train_pos, num_nodes, max_len=10):
    import networkx as nx
    from tqdm import tqdm

    G = nx.Graph()
    G.add_edges_from(train_pos.tolist())
    G.add_nodes_from(range(num_nodes))

    path_tensor = []
    for u, v in tqdm(train_pos.tolist(), desc="Precomputing padded paths"):
        try:
            path = nx.shortest_path(G, u, v)
        except:
            path = [u, v]

        if len(path) < max_len:
            pad = [0] * (max_len - len(path))
            path = path + pad
        else:
            path = path[:max_len]  # truncate

        path_tensor.append(path)

    return torch.tensor(path_tensor, dtype=torch.long)  # shape: [num_edges, max_len]


def preprocess_path_tensor(train_pos, num_nodes, max_len=10):
    import networkx as nx
    from tqdm import tqdm

    G = nx.Graph()
    G.add_edges_from(train_pos.tolist())
    G.add_nodes_from(range(num_nodes))

    path_tensor = []
    for u, v in tqdm(train_pos.tolist(), desc="Precomputing padded paths"):
        try:
            path = nx.shortest_path(G, u, v)
        except:
            path = [u, v]

        if len(path) < max_len:
            pad = [0] * (max_len - len(path))
            path = path + pad
        else:
            path = path[:max_len]  # truncate

        path_tensor.append(path)

    return torch.tensor(path_tensor, dtype=torch.long)  # shape: [num_edges, max_len]

def generate_configs(grid_dict):
    keys = list(grid_dict.keys())
    values = list(itertools.product(*[grid_dict[k] for k in keys]))
    return [dict(zip(keys, v)) for v in values]

def generate_negative_edges(num_nodes, num_samples, G_existing):
    neg_edges = set()
    attempts = 0
    max_attempts = num_samples * 10  # Safety limit to avoid infinite loops

    while len(neg_edges) < num_samples and attempts < max_attempts:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue
        edge = (u, v) if u < v else (v, u)
        if not G_existing.has_edge(*edge):
            neg_edges.add(edge)
        attempts += 1

    return list(neg_edges)

def generate_close_negative_edges(G, num_samples, avoid_edges, max_distance=3):
    neg_edges = set()
    nodes = list(G.nodes)
    pbar = tqdm(total=num_samples, desc="Generating close negative edges")

    while len(neg_edges) < num_samples:
        u = random.choice(nodes)

        # BFS per trovare candidati vicini (entro N hop)
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=max_distance)
        candidates = [v for v in lengths if v != u and not G.has_edge(u, v)]
        if not candidates:
            continue

        v = random.choice(candidates)
        edge = (u, v) if u < v else (v, u)
        if edge not in avoid_edges and edge not in neg_edges:
            neg_edges.add(edge)
            pbar.update(1)

    pbar.close()
    return list(neg_edges)

# Generate negative edges: completely random, avoiding those in G_full
def generate_negative_edges_fast(num_nodes, num_samples, existing_edges):
    neg_edges = set()
    existing_set = set((min(u, v), max(u, v)) for u, v in existing_edges)
    while len(neg_edges) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        if edge not in existing_set and edge not in neg_edges:
            neg_edges.add(edge)
    return list(neg_edges)


def precompute_paths(pairs, G):
    path_dict = {}
    for u, v in tqdm(pairs, desc="Precomputing paths"):
        try:
            path = nx.shortest_path(G, u, v)
        except:
            path = [u, v]
        path_dict[(u, v)] = path
    return path_dict

def compute_train_paths(data, pos_train_edge, save_dir, device):
    positive_edges_list = pos_train_edge.tolist()
    random.shuffle(positive_edges_list)

    # 70% for message passing, 30% for loss
    split_idx = int(0.7 * len(positive_edges_list))
    edges_for_mp = positive_edges_list[:split_idx]
    edges_for_loss = positive_edges_list[split_idx:]

    # Ricostruzione grafi
    G_full = nx.Graph()
    G_full.add_edges_from(positive_edges_list)
    G_full.add_nodes_from(range(data.num_nodes))
    G_message_passing = nx.Graph()
    G_message_passing.add_edges_from(edges_for_mp)
    G_message_passing.add_nodes_from(range(data.num_nodes))

    pos_path_file = os.path.join(save_dir, "pos_paths.pkl")
    neg_path_file = os.path.join(save_dir, "neg_paths.pkl")
    neg_edge_file = os.path.join(save_dir, "neg_edges.pkl")

    if os.path.exists(pos_path_file) and os.path.exists(neg_path_file) and os.path.exists(neg_edge_file):
        with open(pos_path_file, "rb") as f:
            precomputed_pos_paths = pickle.load(f)
        with open(neg_path_file, "rb") as f:
            precomputed_neg_paths = pickle.load(f)
        with open(neg_edge_file, "rb") as f:
            neg_edges_for_loss = pickle.load(f)
    else:
        neg_edges_for_loss = generate_negative_edges_fast(
            num_nodes=data.num_nodes,
            num_samples=len(edges_for_loss),
            existing_edges=G_full.edges)

        precomputed_pos_paths = precompute_paths(edges_for_loss, G_message_passing)
        precomputed_neg_paths = precompute_paths(neg_edges_for_loss, G_message_passing)

        with open(pos_path_file, "wb") as f:
            pickle.dump(precomputed_pos_paths, f)
        with open(neg_path_file, "wb") as f:
            pickle.dump(precomputed_neg_paths, f)
        with open(neg_edge_file, "wb") as f:
            pickle.dump(neg_edges_for_loss, f)

    edge_index = torch.tensor(list(G_message_passing.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
    adj = SparseTensor.from_edge_index(edge_index.to(device), edge_weight, [data.num_nodes, data.num_nodes])

    return edges_for_loss, neg_edges_for_loss, precomputed_pos_paths, precomputed_neg_paths, G_message_passing, adj

def compute_test_paths(data, pos_edge, neg_edge, G_message_passing, save_dir, val_test, device):
    
    pos_edge = pos_edge[:10]
    neg_edge = neg_edge.reshape(-1, 2)[:10*500]
    
    pos_path_file = os.path.join(save_dir, val_test+"_pos_paths.pkl")
    neg_path_file = os.path.join(save_dir, val_test+"_neg_paths.pkl")
    if os.path.exists(pos_path_file) and os.path.exists(neg_path_file):
        with open(pos_path_file, "rb") as f:
            precomputed_pos_paths = pickle.load(f)
        with open(neg_path_file, "rb") as f:
            precomputed_neg_paths = pickle.load(f)
    else:
        precomputed_pos_paths = precompute_paths(pos_edge, G_message_passing)
        precomputed_neg_paths = precompute_paths(neg_edge, G_message_passing)

        with open(pos_path_file, "wb") as f:
            pickle.dump(precomputed_pos_paths, f)
        with open(neg_path_file, "wb") as f:
            pickle.dump(precomputed_neg_paths, f)
    return precomputed_pos_paths, precomputed_neg_paths
    
    
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ##### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    parser.add_argument('--use_hard_negative', default=False, action='store_true')

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)


    args = parser.parse_args()

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print('use_hard_negative: ',args.use_hard_negative)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))
    
    data = dataset[0]

    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes
    split_edge = dataset.get_edge_split()

    if hasattr(data, 'x'):
        if data.x != None:
            x = data.x
            # data.x = data.x.to(torch.float)
            data.x = x.to(device).half()

            if args.cat_n2v_feat:
                print('cat n2v embedding!!')
                n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
                data.x = torch.cat((data.x, n2v_emb), dim=-1)
            
            data.x = data.x.to(device)

        else:

            emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)

    else:
        emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
        input_channel = args.hidden_channels
    
    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.to(torch.float)
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            train_edge_weight = split_edge['train']['weight'].to(device)
            train_edge_weight = train_edge_weight.to(torch.float)
        else:
            train_edge_weight = None

    else:
        train_edge_weight = None

    
    data = T.ToSparseTensor()(data)

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)

        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    
    if args.data_name == 'ogbl-citation2': 
        data.adj_t = data.adj_t.to_symmetric()
        if args.gnn_model == 'GCN':
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t

            
    data = data.to(device)

    config = {
    'gnn_type': 'gcn',
    'gnn_layers': 2,
    'hidden_dim': 32,
    'dropout': 0.5,
    'path_encoder_type': 'transformer',
    'path_encoder_layers': 2,
    'path_encoder_heads': 2,
    'path_hidden_dim': 32,
    'predictor_hidden': 64,
    'input_dim': data.x.size(1)  
    }
    model = build_model_from_config(config).to(device)
    model = model.to(device).half()
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
      
    }

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'

    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'

    if args.data_name == 'ogbl-collab':
        pos_train_edge = split_edge['train']['edge']

        pos_valid_edge = split_edge['valid']['edge']
        
        pos_test_edge = split_edge['test']['edge']
    
        print(args.input_dir)
        
        with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
    
    elif args.data_name == 'ogbl-ppa':
        pos_train_edge = split_edge['train']['edge']
        
        subset_dir = f'{args.input_dir}/{args.data_name}'
        val_pos_ix = torch.load(os.path.join(subset_dir, "valid_samples_index.pt"))
        test_pos_ix = torch.load(os.path.join(subset_dir, "test_samples_index.pt"))

        pos_valid_edge = split_edge['valid']['edge'][val_pos_ix, :]
        pos_test_edge = split_edge['test']['edge'][test_pos_ix, :]

       
        with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
    
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        # neg_valid_edge = split_edge['valid']['target_node_neg'] 

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        # neg_test_edge = split_edge['test']['target_node_neg']

        
        with open(f'{args.input_dir}/{args.data_name}/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'{args.input_dir}/{args.data_name}/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)


    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]

    pos_train_edge = pos_train_edge.to(device)


    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    print('train val val_neg test test_neg: ', pos_train_edge.size(), pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size())
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2    


    edges_for_loss, neg_edges_for_loss, precomputed_pos_paths,precomputed_neg_paths,G_message_passing, adj = compute_train_paths(data, pos_train_edge, args.input_dir+"/"+args.data_name+"/", device)
    val_pos_paths, val_neg_paths = compute_test_paths(data, pos_valid_edge, neg_valid_edge, G_message_passing, args.input_dir+"/"+args.data_name+"/", 'val', device)
    test_pos_paths, test_neg_paths = compute_test_paths(data, pos_test_edge, neg_test_edge, G_message_passing, args.input_dir+"/"+args.data_name+"/", 'test', device)

    train_loader = DataLoader(range(len(edges_for_loss)), batch_size=args.batch_size, shuffle=True)
    from statistics import mean, variance

    best_test_mrrs = []
    best_test_hits20s = []

    for run in range(args.runs):
        print('#################################          ', run, '          #################################')
        
        seed = args.seed if args.runs == 1 else run
        print('seed: ', seed)
        init_seed(seed)

        if emb is not None:
            torch.nn.init.xavier_uniform_(emb.weight)
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(emb.parameters()),
                lr=args.lr,
                weight_decay=args.l2
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.l2
            )

        best_val_mrr = 0
        best_test_metrics = None
        kill_cnt = 0

        for epoch in range(1, args.epochs + 1):
            print(f'--- Epoch {epoch} ---')
            loss = train_path_batch(
                model,
                edges_for_loss,
                neg_edges_for_loss,
                data,
                emb,
                optimizer,
                args.batch_size,
                train_edge_weight,
                precomputed_pos_paths,
                precomputed_neg_paths,
                G_message_passing,
                adj,
                train_loader
            )
            print(f"Train loss: {loss:.4f}")

            pos_val_pred, neg_val_pred = test_path_batch(
                model,
                pos_valid_edge,
                neg_valid_edge,
                data,
                emb,
                adj,
                val_pos_paths,
                val_neg_paths,
                batch_size=args.test_batch_size,
            )

            pos_test_pred, neg_test_pred = test_path_batch(
                model,
                pos_test_edge,
                neg_test_edge,
                data,
                emb,
                adj,
                test_pos_paths,
                test_neg_paths,
                batch_size=args.test_batch_size,
            )

            metrics = get_metric_score(
                evaluator_hit,
                evaluator_mrr,
                pos_train_pred=None,  
                pos_val_pred=pos_val_pred,
                neg_val_pred=neg_val_pred,
                pos_test_pred=pos_test_pred,
                neg_test_pred=neg_test_pred,
            )

            val_mrr = metrics['MRR'][1]
            print("Validation/Test Results:")
            for k, (train_v, val_v, test_v) in metrics.items():
                print(f"{k}: Val={val_v:.4f}, Test={test_v:.4f}")

            # Early stopping logic
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_test_metrics = metrics
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt >= 10:
                    print(f"Early stopping at epoch {epoch} (no improvement in 10 steps)")
                    break

        best_test_mrrs.append(best_test_metrics['MRR'][2])
        best_test_hits20s.append(best_test_metrics['Hits@20'][2])
    mean_mrr = mean(best_test_mrrs)
    var_mrr = variance(best_test_mrrs)
    mean_hits20 = mean(best_test_hits20s)
    var_hits20 = variance(best_test_hits20s)

    print("\n================== FINAL RESULTS OVER SEEDS ==================")
    print(f"MRR: Mean = {mean_mrr:.4f}, Variance = {var_mrr:.6f}")
    print(f"Hits@20: Mean = {mean_hits20:.4f}, Variance = {var_hits20:.6f}")
                
            
            
    


if __name__ == "__main__":

    main()


    