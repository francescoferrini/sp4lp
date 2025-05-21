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
        class TransformerPathEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                encoder_layer = TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=32,
                    batch_first=False  # default
                )
                self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

            def forward(self, path_embeds, padding_mask=None):
                """
                path_embeds: [B, L, D]
                padding_mask: [B, L]
                """

                # Permute path_embeds to [L, B, D]
                path_embeds = path_embeds.permute(1, 0, 2)

                # padding_mask must remain [B, L] — do NOT transpose it
                if padding_mask is not None:
                    assert padding_mask.shape[0] == path_embeds.shape[1], \
                        f"Padding batch mismatch: {padding_mask.shape[0]} vs {path_embeds.shape[1]}"
                    assert padding_mask.shape[1] == path_embeds.shape[0], \
                        f"Padding length mismatch: {padding_mask.shape[1]} vs {path_embeds.shape[0]}"

                out = self.encoder(path_embeds, src_key_padding_mask=padding_mask)  # [L, B, D]
                # print(">> FINAL CHECK:", path_embeds.shape, padding_mask.shape)
                return out[-1]  # [B, D]


        return TransformerPathEncoder()
    elif name == 'transformer2':
        class TransformerPathEncoder2(nn.Module):
            def __init__(self, n_heads=4, dim_feedforward=128, n_layers=2, dropout=0.1, use_positional_encoding=True):
                super().__init__()
                self.use_positional_encoding = use_positional_encoding
                self.embed_dim = embed_dim

                encoder_layer = TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=False  # because we permute
                )
                self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

                if use_positional_encoding:
                    self.pos_embedding = nn.Parameter(torch.randn(512, embed_dim))  # max path length = 512

                self.layer_norm = nn.LayerNorm(embed_dim)

            def forward(self, path_embeds, padding_mask=None):
                # path_embeds: [batch, seq_len, embed_dim]
                path_embeds = path_embeds.permute(1, 0, 2)  # -> [seq_len, batch, embed_dim]

                if self.use_positional_encoding:
                    seq_len = path_embeds.size(0)
                    path_embeds = path_embeds + self.pos_embedding[:seq_len].unsqueeze(1)

                out = self.encoder(path_embeds, src_key_padding_mask=padding_mask)  # [seq_len, batch, embed_dim]

                out = out.permute(1, 0, 2)  # -> [batch, seq_len, embed_dim]

                if padding_mask is not None:
                    mask = ~padding_mask  # True = keep
                    lengths = mask.sum(dim=1).unsqueeze(-1)  # [batch, 1]
                    masked_out = out * mask.unsqueeze(-1)  # [batch, seq_len, embed_dim]
                    pooled = masked_out.sum(dim=1) / lengths.clamp(min=1)
                else:
                    pooled = out.mean(dim=1)

                return self.layer_norm(pooled)
        return TransformerPathEncoder2()
    elif name == 'lstm':
        class LSTMPathEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)

            def forward(self, path_embeds, padding_mask=None):
                lengths = (~padding_mask).sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(path_embeds, lengths, batch_first=True, enforce_sorted=False)
                _, (h_n, _) = self.lstm(packed)
                return h_n[-1]
        return LSTMPathEncoder()
    elif name == 'lstm-bidirectional':
        class LSTMPathEncoderbi(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.bidirectional = True
                self.lstm = nn.LSTM(
                    embed_dim,
                    hidden_dim,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=0.5 if n_layers > 1 else 0.0,
                    bidirectional=self.bidirectional
                )

            def forward(self, path_embeds, padding_mask=None):
                # Calcola le vere lunghezze dei path
                if padding_mask is not None:
                    lengths = (~padding_mask).sum(dim=1).cpu()
                else:
                    lengths = torch.ones(path_embeds.size(0), dtype=torch.long) * path_embeds.size(1)

                # Padded -> Packed sequence
                packed = nn.utils.rnn.pack_padded_sequence(path_embeds, lengths, batch_first=True, enforce_sorted=False)
                _, (h_n, _) = self.lstm(packed)

                # h_n shape: (num_layers * num_directions, batch, hidden_dim)
                # Estrai gli ultimi hidden state di entrambi i lati
                if self.bidirectional:
                    h_fw = h_n[-2]
                    h_bw = h_n[-1]
                    h_out = torch.cat([h_fw, h_bw], dim=-1)  # (batch, 2 * hidden_dim)
                else:
                    h_out = h_n[-1]  # (batch, hidden_dim)

                return h_out
        return LSTMPathEncoderbi()
    else:
        raise ValueError(f"Unsupported path encoder: {name}")

class LinkPredictor(nn.Module):
    def __init__(self, embed_dim, path_embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Nuovo layer MLP
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_u, h_v, h_path):
        h_uv = h_u * h_v
    
        combined = torch.cat([h_uv, h_path], dim=-1) if h_path is not None else h_uv
        return self.mlp(combined).squeeze(-1)

class PathGNNModel(nn.Module):
    def __init__(self, gnn_encoder, path_encoder, predictor, max_path_len=64):
        super().__init__()
        self.encoder = gnn_encoder
        self.path_encoder = path_encoder
        self.predictor = predictor
        self.register_buffer("arange_cache", torch.arange(max_path_len))
    def forward(self, x, edge_index, edge_pairs, path_node_lists):
        h = self.encoder(x, edge_index)

        # Efficient index-based selection
        h_u = torch.index_select(h, 0, edge_pairs[:, 0])
        h_v = torch.index_select(h, 0, edge_pairs[:, 1])

        # Preprocess paths forward and reverse
        all_paths = path_node_lists + [p.flip(0) for p in path_node_lists]
        lengths = torch.tensor([len(p) for p in all_paths], device=h.device)
        flat_indices = torch.cat(all_paths, dim=0)
        flat_embeds = torch.index_select(h, 0, flat_indices)
        embeds_split = torch.split(flat_embeds, lengths.tolist(), dim=0)

        padded_paths = pad_sequence(embeds_split, batch_first=True)
        seq_len = padded_paths.size(1)
        if self.arange_cache.size(0) < seq_len:
            self.arange_cache = torch.arange(seq_len, device=h.device)
        padding_mask = self.arange_cache[:seq_len][None, :] >= lengths[:, None]

        half = padded_paths.size(0) // 2
        padded_paths_1 = padded_paths[:half]     # [B, L, D]
        padded_paths_2 = padded_paths[half:]     # [B, L, D]
        padding_mask_1 = padding_mask[:half]     # [B, L]
        padding_mask_2 = padding_mask[half:]     # [B, L]

        # Passali direttamente
        h_path_1 = self.path_encoder(padded_paths_1, padding_mask_1)
        h_path_2 = self.path_encoder(padded_paths_2, padding_mask_2)

        h_path = h_path_1 + h_path_2
        return self.predictor(h_u, h_v, h_path)
    
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
    


