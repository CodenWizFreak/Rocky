"""
LightGCN Model for Collaborative Filtering

A simplified Graph Convolutional Network for recommendation that only uses
embedding propagation (no feature transformation, no non-linearity).

Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation" (SIGIR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class LightGCN(nn.Module):
    """
    LightGCN model for recommendation.

    Uses simple embedding propagation through graph convolution layers,
    then averages embeddings across all layers for final representation.

    Args:
        num_users: Number of user nodes
        num_items: Number of item (movie) nodes
        emb_dim: Embedding dimension (default: 64)
        n_layers: Number of GCN propagation layers (default: 3)
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        # Learnable embeddings
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def compute_graph_embeddings(self, edge_index: torch.Tensor) -> tuple:
        """
        Propagate embeddings through graph layers using LightGCN convolution.

        The key insight of LightGCN: no feature transformation, no non-linearity.
        Just neighbourhood aggregation with symmetric normalization.

        Args:
            edge_index: [2, E] tensor of (user_local_idx, item_local_idx) edges

        Returns:
            (user_embeddings, item_embeddings) after multi-layer propagation
        """
        # Concatenate user and item embeddings into one big embedding matrix
        # Layout: [user_0, user_1, ..., user_N, item_0, item_1, ..., item_M]
        user_emb = self.user_embedding.weight  # [num_users, emb_dim]
        item_emb = self.item_embedding.weight  # [num_items, emb_dim]
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # [N+M, emb_dim]

        # Build bipartite adjacency: offset item indices by num_users
        src = edge_index[0]                        # user local indices
        dst = edge_index[1] + self.num_users       # item local indices (offset)

        # Bidirectional edges (user→item and item→user)
        full_src = torch.cat([src, dst], dim=0)
        full_dst = torch.cat([dst, src], dim=0)
        full_edge_index = torch.stack([full_src, full_dst], dim=0)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        num_nodes = self.num_users + self.num_items
        deg = degree(full_dst, num_nodes=num_nodes, dtype=all_emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[full_src] * deg_inv_sqrt[full_dst]

        # Multi-layer propagation
        emb_layers = [all_emb]
        current_emb = all_emb

        for _ in range(self.n_layers):
            # Sparse matrix multiplication for aggregation
            # out[j] = sum_i (norm[e] * current_emb[src[e]]) for all edges e where dst[e]=j
            out = torch.zeros_like(current_emb)
            out.index_add_(0, full_dst, current_emb[full_src] * norm.unsqueeze(1))
            current_emb = out
            emb_layers.append(current_emb)

        # Average across all layers (including layer 0 = initial embeddings)
        final_emb = torch.stack(emb_layers, dim=0).mean(dim=0)

        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]

        return user_final, item_final

    def forward(self, edge_index: torch.Tensor, users: torch.Tensor, pos_items: torch.Tensor,
                neg_items: torch.Tensor) -> tuple:
        """
        Forward pass for BPR training.

        Args:
            edge_index: full graph edge index
            users: batch of user indices
            pos_items: batch of positive item indices
            neg_items: batch of negative (sampled) item indices

        Returns:
            (user_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0)
        """
        user_emb_all, item_emb_all = self.compute_graph_embeddings(edge_index)

        user_emb = user_emb_all[users]
        pos_emb = item_emb_all[pos_items]
        neg_emb = item_emb_all[neg_items]

        # Initial embeddings for regularization
        user_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)

        return user_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0

    def get_user_embedding(self, edge_index: torch.Tensor, user_idx: int) -> torch.Tensor:
        """Get the final embedding for a specific user after graph propagation."""
        user_emb_all, _ = self.compute_graph_embeddings(edge_index)
        return user_emb_all[user_idx]

    def get_all_scores(self, edge_index: torch.Tensor, user_idx: int) -> torch.Tensor:
        """Get prediction scores for a user against all items."""
        user_emb_all, item_emb_all = self.compute_graph_embeddings(edge_index)
        user_emb = user_emb_all[user_idx]  # [emb_dim]
        scores = torch.matmul(item_emb_all, user_emb)  # [num_items]
        return scores

    def recommend(self, edge_index: torch.Tensor, user_idx: int,
                  top_k: int = 10, exclude: set = None) -> list:
        """
        Get top-K recommendations for a user.

        Args:
            edge_index: graph edges
            user_idx: local user index
            top_k: number of recommendations
            exclude: set of item indices to exclude (e.g., already watched)

        Returns:
            List of (item_idx, score) tuples
        """
        scores = self.get_all_scores(edge_index, user_idx)

        if exclude:
            for idx in exclude:
                if idx < len(scores):
                    scores[idx] = float('-inf')

        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]


def bpr_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor,
             user_emb_0: torch.Tensor, pos_emb_0: torch.Tensor, neg_emb_0: torch.Tensor,
             reg_weight: float = 1e-4) -> torch.Tensor:
    """
    Bayesian Personalised Ranking (BPR) loss with L2 regularization.

    The idea: a user should score positive items higher than negative items.

    loss = -log(sigmoid(pos_score - neg_score)) + reg * ||embeddings||^2

    Args:
        user_emb: user embeddings after propagation
        pos_emb: positive item embeddings after propagation
        neg_emb: negative item embeddings after propagation
        user_emb_0: initial user embeddings (for regularization)
        pos_emb_0: initial positive item embeddings
        neg_emb_0: initial negative item embeddings
        reg_weight: L2 regularization weight

    Returns:
        scalar loss
    """
    # BPR score difference
    pos_scores = (user_emb * pos_emb).sum(dim=1)   # [batch]
    neg_scores = (user_emb * neg_emb).sum(dim=1)   # [batch]

    bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

    # L2 regularization on initial embeddings
    reg = (1 / 2) * (
        user_emb_0.norm(2).pow(2) +
        pos_emb_0.norm(2).pow(2) +
        neg_emb_0.norm(2).pow(2)
    ) / user_emb_0.shape[0]

    return bpr + reg_weight * reg
