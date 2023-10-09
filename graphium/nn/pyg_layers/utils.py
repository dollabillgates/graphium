import math
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple
from torch import Tensor

from torch_geometric.typing import SparseTensor

from graphium.nn.base_layers import MLP, get_norm
from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch


class PreprocessPositions(nn.Module):
    """
    Compute 3D attention bias and 3D node features according to the 3D position information. 
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        num_kernel,
        in_dim=3,
        num_layers=2,
        activation="gelu",
        first_normalization="none",
    ):
        r"""
        Parameters:
            num_heads:
                Number of attention heads used in self-attention.
            embed_dim:
                Hidden dimension of node features.
            num_kernel:
                Number of gaussian kernels.
            num_layers: The number of layers in the MLP.
            activation: The activation function used in the MLP.
            first_normalization: The normalization function used before the gaussian kernel.

        """
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
            last_layer_is_readout=True,  # Since the output is not proportional to the hidden dim, but to the number of heads
        )

        # make sure the 3D node feature has the same dimension as the embedding size
        # so that it can be added to the original node features
        self.node_proj = nn.Linear(self.num_kernel, self.embed_dim)

    # Batching Function
    def compute_delta_pos_in_batches(pos, batch_size):
        batch, n_node, _ = pos.shape
        delta_pos = torch.zeros((batch, n_node, n_node, 3), device=pos.device, dtype=pos.dtype)
    
        for i in range(0, n_node, batch_size):
            for j in range(0, n_node, batch_size):
                slice_i = slice(i, i+batch_size)
                slice_j = slice(j, j+batch_size)
                delta_pos[:, slice_i, slice_j] = pos[:, slice_i, :].unsqueeze(2) - pos[:, slice_j, :].unsqueeze(1)
    
        return delta_pos
    
    def forward(
        self, batch: Batch, max_num_nodes_per_graph: int, on_ipu: bool, positions_3d_key: str
    ) -> Tuple[Tensor, Tensor]:
        pos = batch[positions_3d_key]
        if self.first_normalization is not None:
            pos = self.first_normalization(pos)
        batch_size = None if pos.device.type != "ipu" else batch.graph_is_true.shape[0]
        pos, mask, idx = to_dense_batch(
            pos,
            batch=batch.batch,
            batch_size=batch_size,
            max_num_nodes_per_graph=max_num_nodes_per_graph,
            drop_nodes_last_graph=on_ipu,
        )
        nan_mask = torch.isnan(pos)[:, 0, 0]
        pos.masked_fill_(nan_mask.unsqueeze(1).unsqueeze(2), 0.0)
        padding_mask = ~mask
        batch, n_node, _ = pos.shape
        batch_size = 1024  # Or any other batch size that fits in your memory
        delta_pos = compute_delta_pos_in_batches(pos, batch_size) # Implement Batching function
        distance = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        distance_feature = self.gaussian(distance)
        attn_bias = self.gaussian_proj(distance_feature)
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()
        attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float("-1000"))
        attn_bias.masked_fill_(nan_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
        distance_feature.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
        distance_feature_sum = distance_feature.sum(dim=-2)
        distance_feature_sum = distance_feature_sum.to(self.node_proj.weight.dtype)
        node_feature = self.node_proj(distance_feature_sum)
        node_feature.masked_fill_(nan_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 0.0)
        node_feature = to_sparse_batch(node_feature, idx)

        return attn_bias, node_feature

class GaussianLayer(nn.Module):
    class GaussianLayer_batch(nn.Module):
    def __init__(self, num_kernels=128, in_dim=3):
        super().__init__()
        self.num_kernels = num_kernels
        self.means = nn.Embedding(1, num_kernels)
        self.stds = nn.Embedding(1, num_kernels)
        nn.init.uniform_(self.means.weight, 0, in_dim)
        nn.init.uniform_(self.stds.weight, 0, in_dim)

    def forward(self, input: Tensor, batch_size=1024) -> Tensor:
        input = input.unsqueeze(-1)
        expanded_input = input.expand(-1, -1, -1, self.num_kernels)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 0.01 
        pre_exp_factor = (2 * math.pi) ** 0.5

        tensor_with_kernel = torch.zeros_like(expanded_input)

        for i in range(0, input.shape[1], batch_size):
            for j in range(0, input.shape[2], batch_size):
                batch_input = expanded_input[:, i:i+batch_size, j:j+batch_size, :]
                batch_kernel = torch.exp(-0.5 * (((batch_input - mean) / std) ** 2)) / (pre_exp_factor * std)
                tensor_with_kernel[:, i:i+batch_size, j:j+batch_size, :] = batch_kernel

        return tensor_with_kernel

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
