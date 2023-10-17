import os
import math
import h5py
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.typing import SparseTensor

from graphium.nn.base_layers import MLP, get_norm
from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch

class PreprocessPositions(nn.Module):
    def __init__(self, num_heads, embed_dim, num_kernel, in_dim=3, num_layers=2, activation="gelu", first_normalization="none"):
        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.first_normalization = get_norm(first_normalization, dim=in_dim)

        self.gaussian = GaussianLayer(self.num_kernel, in_dim=in_dim)
        self.gaussian_proj = MLP(
            in_dim=self.num_kernel,
            hidden_dims=self.num_kernel,
            out_dim=self.num_heads,
            depth=num_layers,
            activation=activation,
            last_layer_is_readout=True,
        )
        self.node_proj = nn.Linear(self.num_kernel, self.embed_dim)

    def forward(self, batch: Batch, max_num_nodes_per_graph: int, on_ipu: bool, positions_3d_key: str) -> Tuple[Tensor, Tensor]:
        pos = batch[positions_3d_key]
        if self.first_normalization is not None:
            pos = self.first_normalization(pos)

        batch_size = 1  # 1024
        attn_bias_list = []
        node_feature_list = []

        pos, mask, idx = to_dense_batch(
            pos,
            batch=batch.batch,
            max_num_nodes_per_graph=max_num_nodes_per_graph,
            drop_nodes_last_graph=on_ipu,
        )

        if torch.isnan(pos).any():
            pos.masked_fill_(torch.isnan(pos)[:, 0, 0].unsqueeze(1).unsqueeze(2), 0.0)

        padding_mask = ~mask

        for graph in range(batch.num_graphs):
            graph_pos = pos[graph]
            nan_mask_graph = torch.isnan(graph_pos)[:, 0]

            n_node = graph_pos.shape[0]
            distance_features_sum = torch.zeros((n_node, self.num_kernel), device=pos.device, dtype=pos.dtype)
            attn_bias_blocks = []

            for i in range(0, n_node, batch_size):
                slice_i = slice(i, min(i + batch_size, n_node))
                delta_pos_batch = graph_pos[slice_i].unsqueeze(1) - graph_pos
                distance = delta_pos_batch.norm(dim=-1)
                distance_features = self.gaussian(distance)
            
                distance_features_sum[slice_i] += distance_features.sum(dim=-2)
                attn_bias_per_head = self.gaussian_proj(distance_features).permute(2, 0, 1)

                attn_bias_blocks.append(attn_bias_per_head.to_sparse())
            
                del distance_features, delta_pos_batch, distance 

            attn_bias = torch.stack(attn_bias_blocks, dim=1)
            node_feature = self.node_proj(distance_features_sum)
            if nan_mask_graph.any():
                attn_bias.masked_fill_(nan_mask_graph.unsqueeze(-1).unsqueeze(-1), 0.0)
                node_feature.masked_fill_(nan_mask_graph.unsqueeze(1), 0.0)

            attn_bias_list.append(attn_bias)
            node_feature_list.append(node_feature)

            del attn_bias, node_feature, nan_mask_graph, distance_features_sum, attn_bias_blocks

        attn_bias = torch.stack(attn_bias_list, dim=0)
        node_feature = torch.cat(node_feature_list, dim=0)
        node_feature = to_sparse_batch(node_feature, idx)

        del attn_bias_list, node_feature_list 

        return attn_bias, node_feature

class GaussianLayer(nn.Module):
    def __init__(self, num_kernels=8, in_dim=3): # num_kernels = 128, 32
        super().__init__()
        self.num_kernels = num_kernels
        self.means = nn.Embedding(1, num_kernels)
        self.stds = nn.Embedding(1, num_kernels)
        nn.init.uniform_(self.means.weight, 0, in_dim)
        nn.init.uniform_(self.stds.weight, 0, in_dim)

    def forward(self, distance: Tensor) -> Tensor:
        distance = distance.unsqueeze(-1)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 0.01

        pre_exp_factor = (2 * math.pi) ** 0.5
        gaussian_kernel = torch.exp(-0.5 * (((distance - mean) / std) ** 2)) / (pre_exp_factor * std)

        return gaussian_kernel

def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Generates triplets from the given edge indices.
        A triplet is defined as a path of length two,
        such that if node A is connected to node B,
        and node B is connected to node C, then there is a triplet (A, B, C).

    Parameters:
        edge_index (LongTensor): The edge indices.
        num_nodes (int): The number of nodes.

    Returns:
        col: The sink node indices of edges from the edge indices.
        row: The source node indices of edges from the edge indices.
        idx_i: The sink node indices of the triplets.
        idx_j: The middle node indices of the triplets.
        idx_k: The source node indices of the triplets.
        idx_kj: The indices of edges those from the source node to the middle node.
        idx_ji: The indices of edges those from the middle node to the sink node.
    """
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # Remove self-loop triplets d->b->d
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k->j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
