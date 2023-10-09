import os
import math
import h5py
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
    def compute_delta_pos_in_batches(self, pos, batch_size):
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
    
        batch_size = 1024  # Or any other batch size that fits in your memory
        attn_bias_list = []
        node_feature_list = []
    
        for graph in range(batch.num_graphs):
            graph_pos = pos[batch.batch == graph]
            nan_mask = torch.isnan(graph_pos)[:, 0]
            graph_pos.masked_fill_(nan_mask.unsqueeze(1), 0.0)
    
            delta_pos = graph_pos.unsqueeze(1) - graph_pos.unsqueeze(0)
            distance = delta_pos.norm(dim=-1).view(1, delta_pos.shape[0], delta_pos.shape[1])
            distance_feature = self.gaussian(distance)
            
            attn_bias = self.gaussian_proj(distance_feature)
            attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()
            # attn_bias.masked_fill_(nan_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
            # Updated masking operation with correction for multiple graphs
            expanded_nan_mask = nan_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(attn_bias.size(0), -1, -1, -1)
            attn_bias.masked_fill_(expanded_nan_mask, 0.0)
    
            distance_feature.masked_fill_(nan_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            distance_feature_sum = distance_feature.sum(dim=-2)
            distance_feature_sum = distance_feature_sum.to(self.node_proj.weight.dtype)
    
            node_feature = self.node_proj(distance_feature_sum)
            node_feature.masked_fill_(nan_mask.unsqueeze(1), 0.0)
    
            attn_bias_list.append(attn_bias)
            node_feature_list.append(node_feature)
    
        attn_bias = torch.cat(attn_bias_list, dim=0)
        node_feature = torch.cat(node_feature_list, dim=0)
        
        return attn_bias, node_feature

class GaussianLayer(nn.Module):
    def __init__(self, num_kernels=32, in_dim=3): # num_kernels = 128
        super().__init__()
        self.num_kernels = num_kernels
        self.means = nn.Embedding(1, num_kernels)
        self.stds = nn.Embedding(1, num_kernels)
        nn.init.uniform_(self.means.weight, 0, in_dim)
        nn.init.uniform_(self.stds.weight, 0, in_dim)

    def forward(self, input: Tensor, batch_size=1024, file_path="tensor_with_kernel.h5") -> Tensor:
        input = input.unsqueeze(-1)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 0.01 
        pre_exp_factor = (2 * math.pi) ** 0.5
    
        with h5py.File(file_path, 'w') as f:
            tensor_with_kernel_dset = f.create_dataset('tensor_with_kernel', 
                                                       shape=(input.shape[0], input.shape[1], input.shape[2], self.num_kernels),
                                                       dtype='f')
    
            for i in range(0, input.shape[1], batch_size):
                for j in range(0, input.shape[2], batch_size):
                    slice_i = slice(i, i + batch_size)
                    slice_j = slice(j, j + batch_size)
                    batch_input = input[:, slice_i, slice_j, :].expand(-1, -1, -1, self.num_kernels)
                    
                    batch_kernel = torch.exp(-0.5 * (((batch_input - mean) / std) ** 2)) / (pre_exp_factor * std)
                    tensor_with_kernel_dset[:, slice_i, slice_j, :] = batch_kernel.cpu().numpy()
    
        # Load data from HDF5 in chunks and reconstruct tensor_with_kernel
        with h5py.File(file_path, 'r') as f:
            tensor_with_kernel_dset = f['tensor_with_kernel']
            tensor_with_kernel = torch.from_numpy(tensor_with_kernel_dset[:])  # Convert this as per your memory handling strategy
            tensor_with_kernel = tensor_with_kernel.to(batch_kernel.device) # Move tensor_with_kernel back to device
    
        os.remove(file_path)  # Clean up the temporary file to save disk space
    
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
