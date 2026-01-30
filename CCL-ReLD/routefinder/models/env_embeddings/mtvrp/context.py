import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = embed_dim * 2
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)
        self.IDT = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.state_proj = nn.Linear(embed_dim * 4, embed_dim, bias=linear_bias)
        self.project_context_b = nn.Linear(3, embed_dim, bias=linear_bias)
        self.project_context_l = nn.Linear(3, embed_dim, bias=linear_bias)
        self.project_context_o = nn.Linear(3, embed_dim, bias=linear_bias)
        self.project_context_tw = nn.Linear(4, embed_dim, bias=linear_bias)
        self.gate_O = nn.Linear(embed_dim, 1)
        self.gate_L = nn.Linear(embed_dim, 1)
        self.gate_TW = nn.Linear(embed_dim, 1)
        self.gate_B = nn.Linear(embed_dim, 1)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        constraint_B, constraint_L, constraint_O, constraint_TW = self._state_embedding(embeddings, td)
        B_embedding = self.project_context_b(torch.nan_to_num(constraint_B, nan=0.0, posinf=0.0, neginf=0.0))
        L_embedding = self.project_context_l(torch.nan_to_num(constraint_L, nan=0.0, posinf=0.0, neginf=0.0))
        O_embedding = self.project_context_o(torch.nan_to_num(constraint_O, nan=0.0, posinf=0.0, neginf=0.0))
        TW_embedding = self.project_context_tw(torch.nan_to_num(constraint_TW, nan=0.0, posinf=0.0, neginf=0.0))
        state_embedding = torch.cat([B_embedding, L_embedding, O_embedding, TW_embedding], -1)
        node_expanded = cur_node_embedding.unsqueeze(0)  # (1, 1024, 50, 128)
        constraint_embedding = torch.stack([B_embedding, L_embedding, O_embedding, TW_embedding])

        # Dot product similarity
        scores = torch.sum(constraint_embedding * node_expanded, dim=-1)  # (4, 1024, 50)
        if self.training:
            noise_std = 0.1

            # Optional: add Gaussian noise
            noise = torch.randn_like(scores) * noise_std
            scores = scores + noise

        # Optional: scale (like transformer)
        d = cur_node_embedding.shape[-1]
        scores = scores / d ** 0.5

        # Apply sigmoid to get gate values independently
        gates = torch.sigmoid(scores)[:, :, :, None]  # (4, 1024, 50)

        # gates = torch.stack([
        #     self.gate_B(B_embedding), self.gate_L(L_embedding), self.gate_O(O_embedding), self.gate_TW(TW_embedding),
        # ])  # shape: [4]

        # 加权求和
        # traj_embedding = gates[0] * B_embedding + gates[1] * L_embedding + gates[2] * O_embedding + gates[3] * TW_embedding

        # if mask is not None:
        curt_msk = td['task_label'].permute(2, 0, 1).unsqueeze(-1).float()
        # curt_msk[0, :, :, :] = 1
        # curt_msk[2, :, :, :] = 1
        gates = gates * curt_msk
        # gates = torch.softmax(gates, dim=0)

        traj_embedding = sum(g * f for g, f in zip(gates, constraint_embedding)) + self.state_proj(state_embedding)
        context_embedding = torch.cat([cur_node_embedding, traj_embedding], -1)
        D_embedding = cur_node_embedding + self.IDT(traj_embedding)
        return torch.stack(
            (self.project_context(context_embedding), B_embedding, L_embedding, O_embedding, TW_embedding, D_embedding))


class MTVRPContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - open route
    - remaining distance (set to default_remain_dist if positive infinity)

    Note that the distance limit (L) and open routes (O) are only embedding during decoding
    in this version
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(MTVRPContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 4
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        curt_node = td['current_node'].unsqueeze(-1)
        d_l = torch.gather(td['demand_linehaul'], 2, curt_node)
        d_b = torch.gather(td['demand_backhaul'], 2, curt_node)
        constraint_B = torch.cat([d_l, d_b, available_load], dim=-1)

        coor = torch.gather(td['locs'], 2, curt_node.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(-2)
        constraint_L = torch.cat([coor, remaining_dist], dim=-1)
        constraint_O = torch.cat([coor, td["current_total_length"]], dim=-1)
        TW = torch.gather(td['time_windows'], 2, curt_node.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(-2)
        TWS = torch.gather(td['service_time'], 2, curt_node)
        constraint_TW = torch.cat([TW, TWS, td["current_time"]], dim=-1)
        # context_feats = torch.cat(
        #     (
        #         available_load,
        #         td["current_time"],
        #         td["open_route"].float(),
        #         remaining_dist,
        #     ),
        #     -1,
        # )
        return constraint_B, constraint_L, constraint_O, constraint_TW


class RouteFinderContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)

    We do not need to embed the open route here since it is done encoder-side.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(RouteFinderContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                remaining_dist,
            ),
            -1,
        )
        return context_feats


# Simple wrapper for better naming. We can replace with this name after getting
# the final models
class MTVRPContextEmbeddingRouteFinder(RouteFinderContextEmbedding):
    def __init__(self, *args, **kwargs):
        super(MTVRPContextEmbeddingRouteFinder, self).__init__(*args, **kwargs)


class MTVRPContextEmbeddingM(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with mixed backhaul.
    This is for the zero-shot or few-short on backhaul_class 2 instances.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingM, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
                                        td["vehicle_capacity"] - td["used_capacity_backhaul"]
                                ) * (td["backhaul_class"] == 2)

        # Note: now we need the projection to have embed_dim + 4 features!
        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
            ),
            -1,
        )


class MTVRPContextEmbeddingFull(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with full features, including the Mixed Backhaul (MB) variants
    and Multi-depot (MD) variants as well.
    In practice, we use the same features as the MTVRPContextEmbeddingRouteFinder, but we add:
    - available load for MB variants (backhaul class 2)
    - locations of the depot where we started the route, since we may need to return there
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4 + 2)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingFull, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
                                        td["vehicle_capacity"] - td["used_capacity_backhaul"]
                                ) * (td["backhaul_class"] == 2)

        start_depot_location = gather_by_index(td["locs"], td["current_depot"], dim=-2)

        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
                start_depot_location,
            ),
            -1,
        )
