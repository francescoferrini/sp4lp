import argparse
from pickle import FALSE

import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv

# from logger import Logger
from torch.nn import Embedding
# from utils import init_seed, get_param
from torch.nn.init import xavier_normal_
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)
from torch_geometric.nn import global_sort_pool
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter_add


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


def get_gnn_layer(name):
    if name == 'gcn':
        return GCNConv
    elif name == 'gat':
        return GATConv
    elif name == 'gin':
        return GINConv
    elif name == 'sage':
        return SAGEConv
    else:
        raise ValueError(f"Unknown GNN type: {name}")

# def get_gnn_encoder(name, in_channels, hidden_channels, out_channels, num_layers, dropout):
#     class GNNEncoder(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.convs = nn.ModuleList()
#             self.dropout = dropout

#             if name == 'gin':
#                 nn_fn = nn.Sequential(
#                     nn.Linear(in_channels, hidden_channels),
#                     nn.ReLU(),
#                     nn.Linear(hidden_channels, hidden_channels)
#                 )
#                 self.convs.append(GINConv(nn_fn))
#             else:
#                 Conv = get_gnn_layer(name)
#                 self.convs.append(Conv(in_channels, hidden_channels))

#             for _ in range(num_layers - 2):
#                 if name == 'gin':
#                     nn_fn = nn.Sequential(
#                         nn.Linear(hidden_channels, hidden_channels),
#                         nn.ReLU(),
#                         nn.Linear(hidden_channels, hidden_channels)
#                     )
#                     self.convs.append(GINConv(nn_fn))
#                 else:
#                     self.convs.append(Conv(hidden_channels, hidden_channels))

#             if name == 'gin':
#                 nn_fn = nn.Sequential(
#                     nn.Linear(hidden_channels, out_channels),
#                     nn.ReLU(),
#                     nn.Linear(out_channels, out_channels)
#                 )
#                 self.convs.append(GINConv(nn_fn))
#             else:
#                 self.convs.append(Conv(hidden_channels, out_channels))

#         def forward(self, x, edge_index):
#             for conv in self.convs[:-1]:
#                 x = conv(x, edge_index)
#                 x = F.relu(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#             x = self.convs[-1](x, edge_index)
#             return x

#     return GNNEncoder()

# def get_path_encoder(name, embed_dim, hidden_dim, n_layers=1, n_heads=4):
#     if name == 'transformer':
#         class TransformerPathEncoder(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 encoder_layer = TransformerEncoderLayer(
#                     d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim
#                 )
#                 self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

#             def forward(self, path_embeds, padding_mask=None):
#                 path_embeds = path_embeds.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
#                 out = self.encoder(path_embeds, src_key_padding_mask=padding_mask)
#                 return out[-1]  # [batch_size, embed_dim]
#         return TransformerPathEncoder()

#     elif name == 'lstm':
#         class LSTMPathEncoder(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)

#             def forward(self, path_embeds, padding_mask=None):
#                 lengths = (~padding_mask).sum(dim=1).cpu()
#                 packed = nn.utils.rnn.pack_padded_sequence(path_embeds, lengths, batch_first=True, enforce_sorted=False)
#                 _, (h_n, _) = self.lstm(packed)
#                 return h_n[-1]
#         return LSTMPathEncoder()

#     else:
#         raise ValueError(f"Unsupported path encoder: {name}")

# class LinkPredictor(nn.Module):
#     def __init__(self, embed_dim, path_embed_dim, hidden_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             # nn.Linear(embed_dim + path_embed_dim, hidden_dim),
#             nn.Linear(embed_dim*2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, h_u, h_v, h_path):
#         h_uv = h_u * h_v
#         combined = torch.cat([h_uv, h_path], dim=-1) if h_path is not None else h_uv
#         return self.mlp(combined).squeeze(-1)

# class PathGNNModel(nn.Module):
#     def __init__(self, gnn_encoder, path_encoder, predictor, max_path_len=64):
#         super().__init__()
#         self.encoder = gnn_encoder
#         self.path_encoder = path_encoder
#         self.predictor = predictor
#         self.register_buffer("arange_cache", torch.arange(max_path_len))
#     # def forward(self, x, edge_index, edge_pairs, path_node_lists):
#     #     h = self.encoder(x, edge_index)

#     #     h_u = torch.index_select(h, 0, edge_pairs[:, 0])
#     #     h_v = torch.index_select(h, 0, edge_pairs[:, 1])

#     #     # Preprocess paths forward and reverse
#     #     all_paths = path_node_lists + [p.flip(0) for p in path_node_lists]
#     #     lengths = torch.tensor([len(p) for p in all_paths], device=h.device)
#     #     flat_indices = torch.cat(all_paths, dim=0)

#     #     # Crea gli indici dei path
#     #     path_ids = torch.arange(len(all_paths), device=h.device)
#     #     path_ids = path_ids.repeat_interleave(lengths)

#     #     # Somma batch-wise con scatter_add
#     #     flat_embeds = h[flat_indices]
#     #     path_sums = scatter_add(flat_embeds, path_ids, dim=0)

#     #     # Combina forward + reverse
#     #     half = len(path_node_lists)
#     #     h_path = path_sums[:half] + path_sums[half:]

#     #     return self.predictor(h_u, h_v, h_path)
#     def forward(self, x, edge_index, edge_pairs, path_node_lists):
#         h = self.encoder(x, edge_index)

#         # Efficient index-based selection
#         h_u = torch.index_select(h, 0, edge_pairs[:, 0])
#         h_v = torch.index_select(h, 0, edge_pairs[:, 1])

#         # Preprocess paths forward and reverse
#         all_paths = path_node_lists + [p.flip(0) for p in path_node_lists]
#         lengths = torch.tensor([len(p) for p in all_paths], device=h.device)
#         flat_indices = torch.cat(all_paths, dim=0)
#         flat_embeds = torch.index_select(h, 0, flat_indices)
#         embeds_split = torch.split(flat_embeds, lengths.tolist(), dim=0)

#         padded_paths = pad_sequence(embeds_split, batch_first=True)
#         seq_len = padded_paths.size(1)
#         if self.arange_cache.size(0) < seq_len:
#             self.arange_cache = torch.arange(seq_len, device=h.device)
#         padding_mask = self.arange_cache[:seq_len][None, :] >= lengths[:, None]

#         h_path_combined = self.path_encoder(padded_paths, padding_mask)
#         half = len(h_path_combined) // 2
#         h_path = h_path_combined[:half] + h_path_combined[half:]
#         return self.predictor(h_u, h_v, h_path)
    
def get_gnn_layer(name):
    if name == 'gcn':
        return GCNConv
    elif name == 'gat':
        return GATConv
    elif name == 'gin':
        return GINConv
    elif name == 'sage':
        return SAGEConv
    else:
        raise ValueError(f"Unknown GNN type: {name}")


def get_gnn_encoder(name, in_channels, hidden_channels, out_channels, num_layers, dropout):
    class GNNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.dropout = dropout

            if name == 'gin':
                nn_fn = nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(nn_fn))
            else:
                Conv = get_gnn_layer(name)
                self.convs.append(Conv(in_channels, hidden_channels))

            for _ in range(num_layers - 2):
                if name == 'gin':
                    nn_fn = nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels)
                    )
                    self.convs.append(GINConv(nn_fn))
                else:
                    self.convs.append(Conv(hidden_channels, hidden_channels))

            if name == 'gin':
                nn_fn = nn.Sequential(
                    nn.Linear(hidden_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
                self.convs.append(GINConv(nn_fn))
            else:
                self.convs.append(Conv(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x

    return GNNEncoder()


def get_path_encoder(name, embed_dim, hidden_dim, n_layers=1, n_heads=4):
    if name == 'transformer':
        # class TransformerPathEncoder(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        #         self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

        #     def forward(self, path_embeds):
        #         path_embeds = path_embeds.permute(1, 0, 2)
        #         out = self.encoder(path_embeds)
        #         return out[-1]

        # return TransformerPathEncoder()
        class TransformerPathEncoder(nn.Module):
            def __init__(self, embed_dim, hidden_dim, n_layers=1, n_heads=4):
                super().__init__()
                encoder_layer = TransformerEncoderLayer(
                    d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim
                )
                self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
            def forward(self, path_embeds, padding_mask=None):
                # path_embeds: [batch_size, seq_len, embed_dim]
                path_embeds = path_embeds.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
                # padding_mask: [batch_size, seq_len]
                out = self.encoder(path_embeds, src_key_padding_mask=padding_mask)
                return out[-1]  # [batch_size, embed_dim]
        return TransformerPathEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
    
    elif name == 'lstm':
        # class LSTMPathEncoder(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)

        #     def forward(self, path_embeds):
        #         _, (h_n, _) = self.lstm(path_embeds)
        #         return h_n[-1]
        class LSTMPathEncoder(nn.Module):
            def __init__(self, embed_dim, hidden_dim, n_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)

            def forward(self, path_embeds, padding_mask=None):
                lengths = (~padding_mask).sum(dim=1).cpu()
                packed = pack_padded_sequence(path_embeds, lengths, batch_first=True, enforce_sorted=False)
                _, (h_n, _) = self.lstm(packed)
                return h_n[-1]

        return LSTMPathEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)

    elif name == 'mamba':
        if not MAMBA_AVAILABLE:
            raise ImportError("You must install mamba-ssm to use MambaPathEncoder")

        class MambaPathEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.mamba = Mamba(d_model=embed_dim, n_layers=n_layers)

            def forward(self, path_embeds):
                return self.mamba(path_embeds)[-1]  # output of last time step

        return MambaPathEncoder()

    else:
        raise ValueError(f"Unsupported path encoder: {name}")


class LinkPredictor(nn.Module):
    def __init__(self, embed_dim, path_embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + path_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_u, h_v, h_path):
        h_uv = h_u * h_v
        if h_path != None:
            combined = torch.cat([h_uv, h_path], dim=-1)
        else:
            combined = h_uv
        return self.mlp(combined).squeeze()


class PathGNNModel(nn.Module):
    def __init__(self, gnn_encoder, path_encoder, predictor, max_path_len=10):
        super().__init__()
        self.encoder = gnn_encoder
        self.path_encoder = path_encoder
        self.predictor = predictor
        # Cache per arange dinamico (utile per mask di padding)
        self.register_buffer("arange_cache", torch.arange(max_path_len))
    def forward(self, x, edge_index, edge_pairs, path_node_lists):
        h = self.encoder(x, edge_index)

        # Uso di index_select invece di slicing per performance
        h_u = torch.index_select(h, 0, edge_pairs[:, 0])
        h_v = torch.index_select(h, 0, edge_pairs[:, 1])

        # Unifica i path forward e reverse in batch
        all_paths = path_node_lists + [p.flip(0) for p in path_node_lists]
        path_lengths = torch.tensor([len(p) for p in all_paths], device=h.device)

        # Flatten dei path in un unico indice
        flat_indices = torch.cat(all_paths, dim=0)
        flat_embeddings = torch.index_select(h, 0, flat_indices)

        # Re-costruisci batch pad-ato (più efficiente che fare [h[path] for path in ...])
        padded_paths = pad_sequence(
            torch.split(flat_embeddings, path_lengths.tolist()), 
            batch_first=True
        )

        # Mask di padding ottimizzata con arange_cache
        seq_len = padded_paths.size(1)
        if self.arange_cache.size(0) < seq_len:
            self.arange_cache = torch.arange(seq_len, device=h.device)
        padding_mask = self.arange_cache[:seq_len][None, :] >= path_lengths[:, None]

        # Unica chiamata al path encoder
        h_path_combined = self.path_encoder(padded_paths, padding_mask)

        # Dividi forward e reverse
        half = len(h_path_combined) // 2
        h_path = h_path_combined[:half] + h_path_combined[half:]

        # Predict
        return self.predictor(h_u, h_v, h_path)
    # def forward(self, x, edge_index, edge_pairs, path_node_lists):
    #     h = self.encoder(x, edge_index)
    #     h_u = h[edge_pairs[:, 0]]
    #     h_v = h[edge_pairs[:, 1]]

    #     path_embeds = [h[path] for path in path_node_lists]
    #     padded = pad_sequence(path_embeds, batch_first=True)
    #     h_path_forward = self.path_encoder(padded)

    #     reverse_embeds = [h[path.flip(0)] for path in path_node_lists]
    #     padded_reverse = pad_sequence(reverse_embeds, batch_first=True)
    #     h_path_reverse = self.path_encoder(padded_reverse)

    #     h_path = h_path_forward + h_path_reverse
    #     return self.predictor(h_u, h_v, h_path)

# Utility to build model from config dict
def build_model_from_config(config):
    gnn = get_gnn_encoder(
        config['gnn_type'],
        config['input_dim'],
        config['hidden_dim'],
        config['hidden_dim'],
        config['gnn_layers'],
        config['dropout']
    )
    path = get_path_encoder(
        config['path_encoder_type'],
        config['hidden_dim'],
        config['path_hidden_dim'],
        config['path_encoder_layers'],
        config.get('path_encoder_heads', 4)
    )
    predictor = LinkPredictor(
        config['hidden_dim'],
        config['hidden_dim'],
        config['predictor_hidden']
    )
    return PathGNNModel(gnn, path, predictor)
    
    
# class GCNEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.dropout = dropout

#         self.convs.append(GCNConv(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))
#         self.convs.append(GCNConv(hidden_channels, out_channels))

#     def forward(self, x, edge_index):
#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x

# class TransformerPathEncoder(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, n_layers=2, n_heads=4):
#         super().__init__()
#         encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim)
#         self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

#     def forward(self, path_embeds):
#         # path_embeds: [B, L, D]
#         path_embeds = path_embeds.permute(1, 0, 2)  # [L, B, D]
#         out = self.encoder(path_embeds)             # [L, B, D]
#         return out[-1]    

# class LinkPredictor(nn.Module):
#     def __init__(self, embed_dim, path_embed_dim, hidden_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim + path_embed_dim, hidden_dim),  # embed_dim da h_u * h_v
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, h_u, h_v, h_path):
#         h_uv = h_u * h_v  # elemento per elemento
#         combined = torch.cat([h_uv, h_path], dim=-1)
#         return self.mlp(combined).squeeze()


# class PathGNNModel(nn.Module):
#     def __init__(self, gcn_args, path_encoder_args, predictor_args):
#         super().__init__()
#         self.encoder = GCNEncoder(*gcn_args)
#         self.path_encoder = TransformerPathEncoder(*path_encoder_args)
#         self.predictor = LinkPredictor(*predictor_args)

#     def forward(self, x, edge_index, edge_pairs, path_node_lists):
#         h = self.encoder(x, edge_index)  # [N, D]
#         h_u = h[edge_pairs[:, 0]]
#         h_v = h[edge_pairs[:, 1]]

#         # Forward path
#         path_embeds = [h[path] for path in path_node_lists]
#         padded = pad_sequence(path_embeds, batch_first=True)  # [B, L, D]
#         h_path_forward = self.path_encoder(padded)             # [B, D]

#         # Reverse path: inverti ogni sequenza
#         reverse_embeds = [h[path.flip(0)] for path in path_node_lists]
#         padded_reverse = pad_sequence(reverse_embeds, batch_first=True)  # [B, L, D]
#         h_path_reverse = self.path_encoder(padded_reverse)               # [B, D]

#         # Somma: forward + reverse
#         h_path = h_path_forward + h_path_reverse  # [B, D]

#         return self.predictor(h_u, h_v, h_path)



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels,normalize=False ))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        
        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if num_layers == 1:
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(in_channels, out_channels, heads=head))

        elif num_layers > 1:
            hidden_channels= int(self.hidden_channels/head)
            self.convs.append(GATConv(in_channels, hidden_channels, heads=head))
            
            for _ in range(num_layers - 2):
                hidden_channels =  int(self.hidden_channels/head)
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels, heads=head))
            
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(hidden_channels, out_channels, heads=head))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gat: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        
        return x



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))

        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in sage: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class mlp_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(mlp_model, self).__init__()

        self.lins = torch.nn.ModuleList()

        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t=None):
        if self.invest == 1:
            print('layers in mlp: ', len(self.lins))
            self.invest = 0
       
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(GIN, self).__init__()

         # self.mlp1= mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
        # self.mlp2 = mlp_model( hidden_channels, hidden_channels, out_channels, gin_mlp_layer, dropout)

        self.convs = torch.nn.ModuleList()
        gin_mlp_layer = mlp_layer
        
        if num_layers == 1:
            self.mlp= mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
            self.convs.append(GINConv(self.mlp))

        else:
            # self.mlp_layers = torch.nn.ModuleList()
            self.mlp1 = mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
            
            self.convs.append(GINConv(self.mlp1))
            for _ in range(num_layers - 2):
                self.mlp = mlp_model( hidden_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
                self.convs.append(GINConv(self.mlp))

            self.mlp2 = mlp_model( hidden_channels, hidden_channels, out_channels, gin_mlp_layer, dropout)
            self.convs.append(GINConv(self.mlp2))

        self.dropout = dropout
        self.invest = 1
          
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # self.mlp1.reset_parameters()
        # self.mlp2.reset_parameters()



    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in gin: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x



class MF(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None, cat_node_feat_mf=False,  data_name=None):
        super(MF, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.data = data_name
        if num_layers == 0:
            out_mf = out_channels
            if self.data=='ogbl-citation2':
                out_mf = 96

            self.emb =  torch.nn.Embedding(node_num, out_mf)
        else:
            self.emb =  torch.nn.Embedding(node_num, in_channels)

        if cat_node_feat_mf:
            in_channels = in_channels*2
    

        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers
        self.cat_node_feat_mf = cat_node_feat_mf

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            
        if self.data == 'ogbl-citation2':
            print('!!!! citaion2 !!!!!')
            torch.nn.init.normal_(self.emb.weight, std = 0.2)

        else: 
            self.emb.reset_parameters()



    def forward(self, x=None, adj_t=None):
        if self.invest == 1:
            print('layers in mlp: ', len(self.lins))
            self.invest = 0
        if self.cat_node_feat_mf and x != None:
            # print('xxxxxxx')
            x = torch.cat((x, self.emb.weight), dim=-1)

        else:
            x =  self.emb.weight


        if self.num_layers == 0:
            return self.emb.weight
        
        else:
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.lins[-1](x)

            return x

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 dynamic_train=False, GNN=GCNConv, use_feature=False, 
                 node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        emb = x

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x



class GCN_seal(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset, 
                 use_feature=False, only_feature=False,node_embedding=None, dropout=0.5):
        super(GCN_seal, self).__init__()
        self.use_feature = use_feature
        self.only_feature = only_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
            
        if self.only_feature:
            initial_channels = train_dataset.num_features

        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        
            
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        tmpx = x
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            if self.invest == 1:
                print('only struct')
            x = z_emb
        if self.only_feature:    ####
            if self.invest == 1:
                print('only feat')
            x = tmpx
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        self.invest = 0
        return x

class SAGE_seal(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, only_feature=False, node_embedding=None, dropout=0.5):
        super(SAGE_seal, self).__init__()
        self.use_feature = use_feature
        self.only_feature = only_feature

        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features

        if self.only_feature:
            initial_channels = train_dataset.num_features


        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        tmpx = x
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
    
        else:
            if self.invest == 1:
                print('only struct')
            x = z_emb
        if self.only_feature:    ####
            if self.invest == 1:
                print('only feat')
            x = tmpx

        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        self.invest = 0
        return x

class DecoupleSEAL(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k, train_dataset, dynamic_train, 
                 node_embedding, dropout, gnn_model):
        super(DecoupleSEAL, self).__init__()
        
        if gnn_model == 'DGCNN':
            self.gnn1 =  DGCNN(hidden_channels, num_layers, max_z, k, train_dataset, 
                 dynamic_train, use_feature=False,  only_feature=False, node_embedding=node_embedding) ###struct


            self.gnn2 =  DGCNN(hidden_channels, num_layers, max_z, k, train_dataset, 
                 dynamic_train, use_feature=False,  only_feature=True, node_embedding=node_embedding) ###feature

        if gnn_model == 'GCN':
            self.gnn1 = GCN_seal(hidden_channels, num_layers, max_z, train_dataset, use_feature=False, only_feature=False,node_embedding=node_embedding, dropout=dropout)  ## structure
            self.gnn2 = GCN_seal(hidden_channels, num_layers, max_z, train_dataset, use_feature=False, only_feature=True, node_embedding=node_embedding, dropout=dropout)  ###feature

        if gnn_model == 'SAGE':
            self.gnn1 = SAGE_seal(hidden_channels, num_layers, max_z, train_dataset, use_feature=False, only_feature=False,node_embedding=node_embedding, dropout=dropout)  ## structure
            self.gnn2 = SAGE_seal(hidden_channels, num_layers, max_z, train_dataset, use_feature=False, only_feature=True, node_embedding=node_embedding, dropout=dropout)  ###feature


        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 0)
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()
    
    def forward(self,z, edge_index, batch, x=None, edge_weight=None, node_id=None):

        logit1 = self.gnn1(z, edge_index, batch, x, edge_weight, node_id)
        logit2 = self.gnn2(z, edge_index, batch, x, edge_weight, node_id)

        alpha = torch.softmax(self.alpha, dim=0)

        scores = alpha[0]*logit1 + alpha[1]*logit2

        return scores
