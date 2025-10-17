import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv
from torch_geometric.transforms import RemoveIsolatedNodes, ToUndirected
from torch_scatter import scatter_add, scatter_max, scatter_mean
from tqdm import tqdm


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _resolve_device()


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_determinism(42)
torch.set_float32_matmul_precision("high")


@dataclass
class ModelConfig:
    d: int = 128
    drop: float = 0.15
    heads: int = 4
    layers: int = 3
    lr: float = 2e-3
    wd: float = 1e-4
    vote_bsz: int = 4096
    bill_bsz: int = 2048
    edge_bsz: int = 8192
    eval_bsz: int = 4096
    time2vec_k: int = 8
    fp16: bool = DEVICE.startswith("cuda")
    backbone: str = "heterosage"
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    delta: float = 0.5
    eta: float = 0.2
    zeta: float = 0.5
    rho: float = 0.01
    tau: float = 0.7
    ls: float = 0.05
    ece_bins: int = 15
    shapley_K: int = 256
    topics_expected: int = 115


def mlp(layers: Iterable[int], final_activation: Optional[nn.Module] = None) -> nn.Sequential:
    seq: List[nn.Module] = []
    dims = list(layers)
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        seq.append(nn.Linear(in_dim, out_dim))
        if out_dim != dims[-1] or final_activation is not None:
            seq.append(nn.ReLU())
    if final_activation is not None:
        seq.append(final_activation)
    return nn.Sequential(*seq)


class Time2Vec(nn.Module):
    def __init__(self, k: int = 8):
        super().__init__()
        self.w0 = nn.Linear(1, 1)
        self.wk = nn.Linear(1, k)

    def forward(self, dt: Tensor) -> Tensor:
        t = dt.view(-1, 1)
        return torch.cat([self.w0(t), torch.sin(self.wk(t))], dim=-1)


class TopicBuilder:
    def __init__(self, expected: int = 115):
        self.expected = expected

    def __call__(self, data: HeteroData) -> Tuple[int, Tensor, Tensor]:
        cluster = data["bill"].cluster.long()
        mask = cluster.ne(-1)
        bill_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        cluster = cluster[mask]
        unique = torch.unique(cluster, sorted=True)
        topics = unique.tolist()
        assert len(topics) == self.expected, f"expected {self.expected} topics got {len(topics)}"
        topic_map = {int(topic): i for i, topic in enumerate(topics)}
        topic_for_bill = torch.full((data["bill"].num_nodes,), -1, dtype=torch.long)
        topic_for_bill[bill_idx] = torch.tensor([topic_map[int(t)] for t in cluster.tolist()], dtype=torch.long)
        edge_index = torch.stack([topic_for_bill[bill_idx], bill_idx], dim=0)
        data["topic"].num_nodes = len(topics)
        data["topic"].x = torch.eye(len(topics))
        data[("topic", "has", "bill")].edge_index = edge_index
        data[("topic", "has", "bill")].edge_attr = torch.ones(edge_index.size(1), 1)
        data["bill"].topic_ix = topic_for_bill
        return len(topics), torch.tensor(topics, dtype=torch.long), topic_for_bill


class FeatureProjector(nn.Module):
    def __init__(self, data: HeteroData, d: int):
        super().__init__()
        dims = {}
        for node_type in data.node_types:
            x = getattr(data[node_type], "x", None)
            dims[node_type] = 0 if x is None else x.size(-1)
        self.projectors = nn.ModuleDict(
            {
                "bill": nn.Linear(max(1, dims.get("bill", 770)), d),
                "bill_version": nn.Linear(max(1, dims.get("bill_version", 390)), d),
                "legislator": nn.Linear(max(1, dims.get("legislator", 385)), d),
                "legislator_term": mlp([max(1, dims.get("legislator_term", 4)), 64, d]),
                "committee": mlp([max(1, dims.get("committee", 65)), 128, d]),
                "lobby_firm": nn.Linear(max(1, dims.get("lobby_firm", 384)), d),
                "donor": mlp([max(1, dims.get("donor", 64)), 128, d]),
            }
        )

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for node_type, x in features.items():
            if node_type in self.projectors:
                out[node_type] = self.projectors[node_type](x)
        return out


class HeteroSAGEBackbone(nn.Module):
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], d: int, layers: int, drop: float, edge_dim: int):
        super().__init__()
        self.edge_mlps = nn.ModuleDict({str(edge_type): mlp([edge_dim, d]) for edge_type in metadata[1]})
        self.convs = nn.ModuleList()
        for _ in range(layers):
            rel_convs = {edge_type: SAGEConv((d, d), d) for edge_type in metadata[1]}
            self.convs.append(HeteroConv(rel_convs, aggr="sum"))
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(d) for node_type in metadata[0]})
        self.drop = nn.Dropout(drop)

    def forward(self, h: Dict[str, Tensor], data: HeteroData, edge_time: Dict[Tuple[str, str, str], Optional[Tensor]]) -> Dict[str, Tensor]:
        for conv in self.convs:
            edge_attr: Dict[Tuple[str, str, str], Optional[Tensor]] = {}
            for edge_type in data.edge_types:
                feats = edge_time.get(edge_type)
                if feats is None:
                    edge_attr[edge_type] = None
                else:
                    edge_attr[edge_type] = self.edge_mlps[str(edge_type)](feats)
            h = conv(h, {edge_type: data[edge_type].edge_index for edge_type in data.edge_types if data[edge_type].edge_index.size(1) > 0}, edge_attr)
            h = {node_type: self.norms[node_type](self.drop(rep)) for node_type, rep in h.items()}
        return h


class HGTBackbone(nn.Module):
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], d: int, layers: int, heads: int, drop: float, edge_dim: int):
        super().__init__()
        self.edge_lin = nn.ModuleDict({str(edge_type): nn.Linear(edge_dim, d) for edge_type in metadata[1]})
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=d,
                    out_channels=d,
                    metadata=metadata,
                    heads=heads,
                    group="sum",
                    dropout=drop,
                    edge_dim=d,
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(d) for node_type in metadata[0]})
        self.drop = nn.Dropout(drop)

    def forward(self, h: Dict[str, Tensor], data: HeteroData, edge_time: Dict[Tuple[str, str, str], Optional[Tensor]]) -> Dict[str, Tensor]:
        edge_attr: Dict[Tuple[str, str, str], Tensor] = {}
        for edge_type in data.edge_types:
            feats = edge_time.get(edge_type)
            lin = self.edge_lin[str(edge_type)]
            if feats is None:
                num_edges = data[edge_type].edge_index.size(1)
                zeros = lin.weight.new_zeros((num_edges, lin.in_features))
                edge_attr[edge_type] = lin(zeros)
            else:
                edge_attr[edge_type] = lin(feats)
        for conv in self.convs:
            h = conv(h, {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}, edge_attr)
            h = {node_type: self.norms[node_type](self.drop(rep)) for node_type, rep in h.items()}
        return h


class MetaPathAggregator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fuse = mlp([5 * d, d])

    def _scatter_mean(self, src: Tensor, index: Tensor, dim_size: int) -> Tensor:
        return scatter_mean(src, index, dim=0, dim_size=dim_size)

    def _gather(self, tensor: Tensor, index: Tensor) -> Tensor:
        return tensor.index_select(0, index)

    def forward(
        self,
        batch: HeteroData,
        h: Dict[str, Tensor],
        vote_edges: Tensor,
        topic_for_bill: Tensor,
    ) -> Tensor:
        lt_idx, bv_idx = vote_edges
        bv2b = batch[("bill_version", "is_version", "bill")].edge_index[1]
        bill_idx = bv2b[bv_idx]
        topic_idx = topic_for_bill[bill_idx]

        lt_vote_pool = self._scatter_mean(h["legislator_term"], lt_idx, h["legislator_term"].size(0))
        lt_vote_context = self._gather(lt_vote_pool, lt_idx)

        bv_prior_edge = batch.get(("bill_version", "priorVersion", "bill_version"))
        if bv_prior_edge is not None and bv_prior_edge.edge_index.numel() > 0:
            prior_ctx = self._scatter_mean(h["bill_version"], bv_prior_edge.edge_index[0], h["bill_version"].size(0))
            prior_ctx = self._gather(prior_ctx, bv_idx)
        else:
            prior_ctx = h["bill_version"].new_zeros((bv_idx.size(0), h["bill_version"].size(1)))

        read_edge = batch.get(("bill_version", "read", "committee"))
        if read_edge is not None and read_edge.edge_index.numel() > 0:
            committee_ctx = self._scatter_mean(h["committee"], read_edge.edge_index[1], h["committee"].size(0))
            committee_ctx = committee_ctx.index_select(0, bill_idx)
        else:
            committee_ctx = h["committee"].new_zeros((bv_idx.size(0), h["committee"].size(1)))

        member_edge = batch.get(("legislator_term", "member_of", "committee"))
        if member_edge is not None and member_edge.edge_index.numel() > 0:
            lt_committee = self._scatter_mean(h["committee"], member_edge.edge_index[1], h["committee"].size(0))
            lt_committee = lt_committee.index_select(0, lt_idx)
        else:
            lt_committee = h["committee"].new_zeros((lt_idx.size(0), h["committee"].size(1)))

        topic_ctx = h["topic"].index_select(0, topic_idx.clamp(min=0))
        fused = torch.cat([lt_vote_context, prior_ctx, committee_ctx, lt_committee, topic_ctx], dim=-1)
        return self.fuse(fused)


class CrossAttention(nn.Module):
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        q = self.q(query).view(-1, self.h, self.dk)
        k = self.k(context).view(-1, self.h, self.dk)
        v = self.v(context).view(-1, self.h, self.dk)
        attn = torch.softmax((q * k).sum(-1) / math.sqrt(self.dk), dim=-1).unsqueeze(-1)
        return self.out((attn * v).reshape(query.size(0), -1))


class VoteHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = mlp([4 * d, 2 * d, d])
        self.out = nn.Linear(d, 3)

    def forward(self, lt: Tensor, bv: Tensor, topic: Tensor, ctx: Tensor) -> Tensor:
        return self.out(self.net(torch.cat([lt, bv, topic, ctx], dim=-1)))


class OutcomeHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = mlp([3 * d, d])
        self.out = nn.Linear(d, 3)

    def forward(self, bill: Tensor, committee_ctx: Tensor, vote_margin: Tensor) -> Tensor:
        return self.out(self.net(torch.cat([bill, committee_ctx, vote_margin], dim=-1)))


class GateHead(nn.Module):
    def __init__(self, d: int, stages: int = 6):
        super().__init__()
        self.net = mlp([3 * d, d])
        self.out = nn.Linear(d, stages)

    def forward(self, committee: Tensor, bill: Tensor, topic: Tensor) -> Tensor:
        return self.out(self.net(torch.cat([committee, bill, topic], dim=-1)))


class StanceHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = mlp([2 * d, d])
        self.out = nn.Linear(d, 1)

    def forward(self, actor: Tensor, topic: Tensor) -> Tensor:
        return torch.tanh(self.out(self.net(torch.cat([actor, topic], dim=-1))))


class MaskNet(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = mlp([d, d, 1])

    def forward(self, x: Tensor, temperature: float = 0.5, training: bool = True) -> Tensor:
        logits = self.net(x).squeeze(-1)
        if training:
            noise = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(noise.clamp_min(1e-6)))
            logits = (logits + gumbel) / temperature
        return torch.sigmoid(logits)


def edge_time_features(data: HeteroData, module: Time2Vec, device: torch.device) -> Dict[Tuple[str, str, str], Optional[Tensor]]:
    feats: Dict[Tuple[str, str, str], Optional[Tensor]] = {}
    for edge_type in data.edge_types:
        store = data[edge_type]
        dt: Optional[Tensor] = None
        edge_time = getattr(store, "edge_time", None)
        if edge_time is not None:
            dt = edge_time.to(device).float()
        else:
            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    dt = edge_attr.to(device).float()
                elif edge_attr.size(-1) > 0:
                    dt = edge_attr[:, -1].to(device).float()
        if dt is None:
            # Some relation types do not track timestamps/attributes; return None so downstream
            # layers avoid constructing artificial edge features that break SAGEConv.
            feats[edge_type] = None
            continue
        feats[edge_type] = module(dt)
    return feats


def attentive_version_pool(batch: HeteroData, h: Dict[str, Tensor]) -> Tensor:
    edge = batch.get(("bill_version", "is_version", "bill"))
    if edge is None or edge.edge_index.size(1) == 0:
        return h["bill"]
    src, dst = edge.edge_index
    weights = scatter_mean(h["bill_version"], src, dim=0, dim_size=h["bill_version"].size(0))
    pooled = scatter_mean(weights[src], dst, dim=0, dim_size=h["bill"].size(0))
    return 0.5 * h["bill"] + 0.5 * pooled


def balanced_ce(logits: Tensor, target: Tensor, num_classes: int = 3, label_smoothing: float = 0.0) -> Tensor:
    mask = target.ge(0)
    if mask.sum() == 0:
        return logits.sum() * 0
    target = target[mask].long()
    logits = logits[mask]
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        counts = torch.bincount(target, minlength=num_classes).float().clamp_min(1)
        weights = (counts.sum() / counts)
        weights = weights / weights.mean()
    one_hot = F.one_hot(target, num_classes=num_classes).float()
    if label_smoothing > 0:
        one_hot = (1 - label_smoothing) * one_hot + label_smoothing / num_classes
    loss = -(weights[target] * (one_hot * log_probs).sum(dim=-1)).mean()
    return loss


def brier_score(logits: Tensor, target: Tensor, num_classes: int = 3) -> Tensor:
    mask = target.ge(0)
    if mask.sum() == 0:
        return logits.sum() * 0
    probs = F.softmax(logits[mask], dim=-1)
    target_one_hot = F.one_hot(target[mask].long(), num_classes=num_classes).float()
    return ((probs - target_one_hot) ** 2).mean()


def mse_with_mask(pred: Tensor, target: Tensor) -> Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0
    return F.mse_loss(pred[mask], target[mask])


def pairwise_rank_loss(scores: Tensor, target: Tensor) -> Tensor:
    mask = torch.isfinite(target)
    if mask.sum() < 2:
        return scores.sum() * 0
    valid_scores = scores[mask]
    valid_target = target[mask]
    diff = valid_target.unsqueeze(1) - valid_target.unsqueeze(0)
    sign = torch.sign(diff)
    pred_diff = valid_scores.unsqueeze(1) - valid_scores.unsqueeze(0)
    loss = F.relu(1 - sign * pred_diff)
    return loss.mean()


def orthogonality_penalty(embeddings: Tensor) -> Tensor:
    normed = F.normalize(embeddings, dim=-1)
    gram = normed @ normed.t()
    identity = torch.eye(gram.size(0), device=embeddings.device)
    return ((gram - identity) ** 2).mean()


def expected_calibration_error(logits: Tensor, target: Tensor, bins: int = 15) -> Tensor:
    mask = target.ge(0)
    logits = logits[mask]
    target = target[mask]
    if target.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(target)
    ece = torch.tensor(0.0, device=logits.device)
    for idx in range(bins):
        lower, upper = idx / bins, (idx + 1) / bins
        mask_bin = (confidences >= lower) & (confidences < upper)
        if mask_bin.any():
            ece += mask_bin.float().mean() * (correct[mask_bin].float().mean() - confidences[mask_bin].mean()).abs()
    return ece


class InfluenceEstimator:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def vote_effect(self, probs: Tensor, edge_index: Tensor, bill_index: Tensor) -> Tensor:
        yes = probs[:, 2]
        no = probs[:, 0]
        margin = scatter_add(yes - no, bill_index, dim=0)
        return margin

    def shapley_vote(self, probs: Tensor, lt_idx: Tensor, bill_idx: Tensor, num_legislators: int, num_bills: int) -> Tensor:
        yes = probs[:, 2]
        pivots = torch.zeros((num_legislators, num_bills), device=probs.device)
        for _ in range(self.cfg.shapley_K):
            perm = torch.randperm(lt_idx.size(0), device=probs.device)
            ordered_lt = lt_idx[perm]
            ordered_bill = bill_idx[perm]
            ordered_yes = yes[perm]
            cumulative = torch.zeros(num_bills, device=probs.device)
            for lt, b, vote in zip(ordered_lt, ordered_bill, ordered_yes):
                before = torch.sigmoid(cumulative[b])
                after = torch.sigmoid(cumulative[b] + vote)
                pivots[lt, b] += after - before
                cumulative[b] += vote
        return pivots / self.cfg.shapley_K

    def ablation_effect(self, probs: Tensor, masks: Tensor, bill_idx: Tensor, num_bills: int) -> Tensor:
        yes = probs[:, 2]
        masked_yes = yes * (1 - masks)
        delta = scatter_add(yes, bill_idx, dim=0) - scatter_add(masked_yes, bill_idx, dim=0)
        return delta / num_bills


class LeGNN4(nn.Module):
    def __init__(self, data: HeteroData, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.data = data
        self.proj = FeatureProjector(data, cfg.d)
        self.topic_embeddings = nn.Embedding(data["topic"].num_nodes, cfg.d)
        self.time_encoder = Time2Vec(cfg.time2vec_k)
        metadata = data.metadata()
        edge_dim = cfg.time2vec_k + 1
        if cfg.backbone == "hgt":
            self.backbone: nn.Module = HGTBackbone(metadata, cfg.d, cfg.layers, cfg.heads, cfg.drop, edge_dim)
        else:
            self.backbone = HeteroSAGEBackbone(metadata, cfg.d, cfg.layers, cfg.drop, edge_dim)
        self.metapath = MetaPathAggregator(cfg.d)
        self.cross_attention = CrossAttention(cfg.d, heads=cfg.heads)
        self.vote_head = VoteHead(cfg.d)
        self.outcome_head = OutcomeHead(cfg.d)
        self.gate_head = GateHead(cfg.d)
        self.stance_lt = StanceHead(cfg.d)
        self.stance_donor = StanceHead(cfg.d)
        self.stance_lobby = StanceHead(cfg.d)
        self.actor_mask = MaskNet(cfg.d)
        self.committee_mask = MaskNet(cfg.d)

    def encode(self, batch: HeteroData) -> Dict[str, Tensor]:
        features: Dict[str, Tensor] = {}
        for node_type in batch.node_types:
            x = getattr(batch[node_type], "x", None)
            if x is None:
                if node_type == "topic":
                    features[node_type] = self.topic_embeddings.weight
                else:
                    features[node_type] = torch.zeros((batch[node_type].num_nodes, self.cfg.d), device=batch[node_type].edge_index.device if hasattr(batch[node_type], "edge_index") else torch.device(DEVICE))
            else:
                features[node_type] = x.to(batch[node_type].x.device)
        projected = self.proj({k: v for k, v in features.items() if k != "topic"})
        projected["topic"] = self.topic_embeddings.weight
        edge_time = edge_time_features(batch, self.time_encoder, projected["bill"].device)
        h = self.backbone(projected, batch, edge_time)
        h["bill"] = attentive_version_pool(batch, h)
        return h

    def vote_forward(self, batch: HeteroData, h: Dict[str, Tensor], topic_for_bill: Tensor) -> Tensor:
        e = batch[("legislator_term", "voted_on", "bill_version")]
        edge_index = e.edge_index
        lt_idx, bv_idx = edge_index
        bv2b = batch[("bill_version", "is_version", "bill")].edge_index[1]
        bill_idx = bv2b[bv_idx]
        topic_idx = topic_for_bill[bill_idx].clamp(min=0)
        ctx = self.metapath(batch, h, edge_index, topic_for_bill)
        attn = self.cross_attention(h["legislator_term"].index_select(0, lt_idx), h["bill"].index_select(0, bill_idx))
        logits = self.vote_head(
            h["legislator_term"].index_select(0, lt_idx),
            h["bill_version"].index_select(0, bv_idx),
            h["topic"].index_select(0, topic_idx),
            ctx + attn,
        )
        return logits

    def outcome_forward(self, batch: HeteroData, h: Dict[str, Tensor], vote_margin: Tensor, topic_for_bill: Tensor) -> Tensor:
        bill_topic = h["topic"].index_select(0, topic_for_bill.clamp(min=0))
        read_edge = batch.get(("bill_version", "read", "committee"))
        if read_edge is not None and read_edge.edge_index.numel() > 0:
            committee_ctx = scatter_mean(
                h["committee"].index_select(0, read_edge.edge_index[1]),
                read_edge.edge_index[0],
                dim=0,
                dim_size=h["bill_version"].size(0),
            )
            committee_ctx = scatter_mean(committee_ctx, batch[("bill_version", "is_version", "bill")].edge_index[1], dim=0, dim_size=h["bill"].size(0))
        else:
            committee_ctx = h["bill"].new_zeros(h["bill"].shape)
        margin = vote_margin.index_select(0, torch.arange(h["bill"].size(0), device=h["bill"].device))
        return self.outcome_head(h["bill"], committee_ctx + bill_topic, margin)

    def gate_forward(self, batch: HeteroData, h: Dict[str, Tensor], topic_for_bill: Tensor) -> Tensor:
        edge = batch[("bill_version", "read", "committee")]
        bv_idx, committee_idx = edge.edge_index
        bill_idx = batch[("bill_version", "is_version", "bill")].edge_index[1][bv_idx]
        topic_idx = topic_for_bill[bill_idx].clamp(min=0)
        return self.gate_head(
            h["committee"].index_select(0, committee_idx),
            h["bill"].index_select(0, bill_idx),
            h["topic"].index_select(0, topic_idx),
        )

    def stance_forward(self, h: Dict[str, Tensor], actor_type: str, actor_idx: Tensor, topic_idx: Tensor) -> Tensor:
        topic = h["topic"].index_select(0, topic_idx)
        if actor_type == "legislator_term":
            return self.stance_lt(h[actor_type].index_select(0, actor_idx), topic)
        if actor_type == "donor":
            return self.stance_donor(h[actor_type].index_select(0, actor_idx), topic)
        if actor_type == "lobby_firm":
            return self.stance_lobby(h[actor_type].index_select(0, actor_idx), topic)
        raise ValueError(actor_type)

    def actor_masks(self, representations: Tensor, training: bool = True) -> Tensor:
        return self.actor_mask(representations, training=training)

    def committee_masks(self, representations: Tensor, training: bool = True) -> Tensor:
        return self.committee_mask(representations, training=training)


def build_neighbor_loaders(data: HeteroData, cfg: ModelConfig) -> Tuple[NeighborLoader, NeighborLoader, LinkNeighborLoader]:
    vote_neighbors = {
        ("legislator_term", "voted_on", "bill_version"): [64, 64, 64],
        ("bill_version", "rev_voted_on", "legislator_term"): [0, 0, 0],
        ("bill_version", "is_version", "bill"): [10, 10, 10],
        ("bill", "rev_is_version", "bill_version"): [0, 0, 0],
        ("bill_version", "priorVersion", "bill_version"): [4, 4, 4],
        ("bill_version", "rev_priorVersion", "bill_version"): [0, 0, 0],
        ("bill_version", "read", "committee"): [6, 6, 6],
        ("committee", "rev_read", "bill_version"): [0, 0, 0],
        ("legislator_term", "member_of", "committee"): [8, 8, 8],
        ("committee", "rev_member_of", "legislator_term"): [0, 0, 0],
        ("legislator", "samePerson", "legislator_term"): [4, 4, 4],
        ("legislator_term", "rev_samePerson", "legislator"): [0, 0, 0],
        ("topic", "has", "bill"): [16, 16, 16],
        ("legislator_term", "wrote", "bill_version"): [6, 6, 6],
        ("bill_version", "rev_wrote", "legislator_term"): [0, 0, 0],
        ("donor", "donated_to", "legislator_term"): [16, 16, 16],
        ("legislator_term", "rev_donated_to", "donor"): [0, 0, 0],
        ("lobby_firm", "lobbied", "legislator_term"): [16, 16, 16],
        ("legislator_term", "rev_lobbied", "lobby_firm"): [0, 0, 0],
        ("lobby_firm", "lobbied", "committee"): [16, 16, 16],
        ("committee", "rev_lobbied", "lobby_firm"): [0, 0, 0],
    }
    vote_loader = NeighborLoader(
        data,
        input_nodes=("legislator_term", torch.arange(data["legislator_term"].num_nodes)),
        num_neighbors=vote_neighbors,
        batch_size=cfg.vote_bsz,
        shuffle=True,
        num_workers=max(2, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
    )

    bill_neighbors = {
        ("bill_version", "is_version", "bill"): [10, 10, 10],
        ("bill_version", "read", "committee"): [6, 6, 6],
        ("legislator_term", "voted_on", "bill_version"): [32, 32, 32],
        ("topic", "has", "bill"): [12, 12, 12],
        ("legislator_term", "wrote", "bill_version"): [6, 6, 6],
    }
    bill_loader = NeighborLoader(
        data,
        input_nodes=("bill", torch.arange(data["bill"].num_nodes)),
        num_neighbors=bill_neighbors,
        batch_size=cfg.bill_bsz,
        shuffle=True,
        num_workers=max(2, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
    )

    gate_loader = LinkNeighborLoader(
        data,
        edge_label_index=("bill_version", "read", "committee"),
        num_neighbors={
            ("bill_version", "is_version", "bill"): [8, 8, 8],
            ("legislator_term", "voted_on", "bill_version"): [16, 16, 16],
            ("topic", "has", "bill"): [8, 8, 8],
            ("legislator_term", "member_of", "committee"): [8, 8, 8],
        },
        batch_size=cfg.edge_bsz,
        shuffle=True,
        num_workers=max(2, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
    )
    return vote_loader, bill_loader, gate_loader


def build_legislator_topic_labels(data: HeteroData, topic_for_bill: Tensor, min_votes: int = 5) -> Tensor:
    vote_edge = data[("legislator_term", "voted_on", "bill_version")]
    labels = vote_edge.edge_attr[:, 0].long()
    mask = labels.ne(0)
    bv2b = data[("bill_version", "is_version", "bill")].edge_index[1]
    bill_idx = bv2b[vote_edge.edge_index[1]]
    topic_idx = topic_for_bill[bill_idx]
    mask &= topic_idx.ge(0)
    lt_idx = vote_edge.edge_index[0][mask]
    topic_idx = topic_idx[mask]
    labels = labels[mask].float()
    num_topics = topic_for_bill.clamp(min=0).max().item() + 1
    stance = torch.zeros(data["legislator_term"].num_nodes, num_topics)
    counts = torch.zeros_like(stance)
    index = lt_idx * num_topics + topic_idx
    scatter_add(labels, index, out=stance.view(-1))
    scatter_add(torch.ones_like(labels), index, out=counts.view(-1))
    stance = stance.view_as(counts) / counts.clamp_min(1)
    stance[counts < min_votes] = float("nan")
    return stance


class Trainer:
    def __init__(self, data: HeteroData, cfg: ModelConfig):
        self.cfg = cfg
        base = data.clone()
        base = ToUndirected()(base)
        base = RemoveIsolatedNodes()(base)
        for edge_type in base.edge_types:
            edge_index = base[edge_type].edge_index
            if isinstance(edge_index, torch.Tensor) and edge_index.dtype != torch.long:
                base[edge_type].edge_index = edge_index.long()
        self.topic_builder = TopicBuilder(cfg.topics_expected)
        self.num_topics, self.topic_values, self.topic_for_bill = self.topic_builder(base)
        self.data = base
        self.model = LeGNN4(self.data, cfg).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.vote_loader, self.bill_loader, self.gate_loader = build_neighbor_loaders(self.data, cfg)
        self.stance_labels = build_legislator_topic_labels(self.data, self.topic_for_bill)
        self.influence = InfluenceEstimator(cfg)

    def _vote_step(self, batch: HeteroData) -> Dict[str, float]:
        batch = batch.to(DEVICE, non_blocking=True)
        h = self.model.encode(batch)
        logits = self.model.vote_forward(batch, h, self.topic_for_bill.to(DEVICE))
        edge = batch[("legislator_term", "voted_on", "bill_version")]
        target = edge.edge_attr[:, 0].to(DEVICE).long()
        loss_vote = balanced_ce(logits, target, label_smoothing=self.cfg.ls) + 0.1 * brier_score(logits, target)
        probs = F.softmax(logits, dim=-1)
        lt_idx, bv_idx = edge.edge_index
        bv2b = batch[("bill_version", "is_version", "bill")].edge_index[1]
        bill_idx = bv2b[bv_idx]
        vote_margin = self.influence.vote_effect(probs, edge.edge_index, bill_idx)
        mask_vals = self.model.actor_masks(h["legislator_term"].index_select(0, lt_idx), self.model.training)
        loss_influence = (mask_vals * (probs[:, 2] - probs[:, 0]).abs()).mean()
        con_loss = 0.5 * (1 - F.cosine_similarity(h["bill"].index_select(0, bill_idx), h["topic"].index_select(0, self.topic_for_bill.to(DEVICE)[bill_idx]))).mean()
        loss = self.cfg.alpha * loss_vote + self.cfg.eta * con_loss + self.cfg.zeta * loss_influence + self.cfg.rho * orthogonality_penalty(self.model.topic_embeddings.weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            ece = expected_calibration_error(logits, target, bins=self.cfg.ece_bins)
            f1 = macro_f1(logits, target)
        return {
            "vote": float(loss_vote.item()),
            "contrast": float(con_loss.item()),
            "influence": float(loss_influence.item()),
            "reg": float(orthogonality_penalty(self.model.topic_embeddings.weight).item()),
            "vote_ece": float(ece.item()),
            "vote_f1": float(f1.item()),
            "vote_margin": vote_margin.detach().mean().item(),
        }

    def _outcome_step(self, batch: HeteroData, vote_margin: Tensor) -> Dict[str, float]:
        batch = batch.to(DEVICE, non_blocking=True)
        h = self.model.encode(batch)
        logits = self.model.outcome_forward(batch, h, vote_margin.to(DEVICE), self.topic_for_bill.to(DEVICE))
        target = batch["bill"].y.to(DEVICE).long()
        loss_out = balanced_ce(logits, target, label_smoothing=self.cfg.ls) + 0.1 * brier_score(logits, target)
        self.optimizer.zero_grad()
        loss_out.backward()
        self.optimizer.step()
        with torch.no_grad():
            ece = expected_calibration_error(logits, target, bins=self.cfg.ece_bins)
            f1 = macro_f1(logits, target)
        return {"out": float(loss_out.item()), "out_ece": float(ece.item()), "out_f1": float(f1.item())}

    def _gate_step(self, batch: HeteroData) -> Dict[str, float]:
        batch = batch.to(DEVICE, non_blocking=True)
        h = self.model.encode(batch)
        logits = self.model.gate_forward(batch, h, self.topic_for_bill.to(DEVICE))
        target = batch.edge_label.to(DEVICE).long().clamp(0, 5)
        loss_gate = balanced_ce(logits, target, num_classes=6, label_smoothing=self.cfg.ls)
        self.optimizer.zero_grad()
        loss_gate.backward()
        self.optimizer.step()
        return {"gate": float(loss_gate.item())}

    def _stance_step(self, h: Dict[str, Tensor]) -> Dict[str, float]:
        stats = {"stance": 0.0, "rank": 0.0}
        if self.stance_labels.numel() == 0:
            return stats
        num_lt, num_topics = self.stance_labels.shape
        lt_idx = torch.randint(0, num_lt, (self.cfg.eval_bsz,), device=DEVICE)
        topic_idx = torch.randint(0, num_topics, (self.cfg.eval_bsz,), device=DEVICE)
        target = self.stance_labels[lt_idx.cpu(), topic_idx.cpu()].to(DEVICE)
        pred = self.model.stance_forward(h, "legislator_term", lt_idx, topic_idx).squeeze(-1)
        loss_stance = mse_with_mask(pred, target)
        loss_rank = pairwise_rank_loss(pred, target)
        self.optimizer.zero_grad()
        (self.cfg.delta * loss_stance + 0.1 * loss_rank).backward()
        self.optimizer.step()
        stats["stance"] = float(loss_stance.item())
        stats["rank"] = float(loss_rank.item())
        return stats

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        agg = {}
        vote_margin_cache = torch.zeros(self.data["bill"].num_nodes)
        for batch in tqdm(self.vote_loader, desc="vote"):
            stats = self._vote_step(batch)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
        vote_margin_cache.fill_(stats.get("vote_margin", 0.0))
        for batch in tqdm(self.bill_loader, desc="outcome"):
            stats = self._outcome_step(batch, vote_margin_cache)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
        for batch in tqdm(self.gate_loader, desc="gate"):
            stats = self._gate_step(batch)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
        full = self.data.to(DEVICE, non_blocking=True)
        h = self.model.encode(full)
        stats = self._stance_step(h)
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v
        return agg

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        vote_metrics = {"f1": 0.0, "ece": 0.0}
        count = 0
        for batch in self.vote_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            h = self.model.encode(batch)
            edge = batch[("legislator_term", "voted_on", "bill_version")]
            logits = self.model.vote_forward(batch, h, self.topic_for_bill.to(DEVICE))
            target = edge.edge_attr[:, 0].to(DEVICE).long()
            vote_metrics["f1"] += float(macro_f1(logits, target).item())
            vote_metrics["ece"] += float(expected_calibration_error(logits, target, bins=self.cfg.ece_bins).item())
            count += 1
            if count >= 5:
                break
        for key in vote_metrics:
            vote_metrics[key] /= max(count, 1)
        return vote_metrics

    @torch.no_grad()
    def embed_full(self) -> Dict[str, Tensor]:
        self.model.eval()
        full = self.data.to(DEVICE, non_blocking=True)
        return self.model.encode(full)

    @torch.no_grad()
    def compute_influence(self, h: Dict[str, Tensor]) -> Dict[str, List[Dict[str, object]]]:
        vote_edge = self.data[("legislator_term", "voted_on", "bill_version")]
        lt_idx, bv_idx = vote_edge.edge_index
        bv2b = self.data[("bill_version", "is_version", "bill")].edge_index[1]
        bill_idx = bv2b[bv_idx]
        topic_idx = self.topic_for_bill[bill_idx]
        batch = self.data.to(DEVICE)
        logits = self.model.vote_forward(batch, h, self.topic_for_bill.to(DEVICE))
        probs = F.softmax(logits, dim=-1)
        masks = self.model.actor_masks(h["legislator_term"].index_select(0, lt_idx), training=False)
        delta = self.influence.ablation_effect(probs, masks, bill_idx.to(DEVICE), self.data["bill"].num_nodes)
        shapley = self.influence.shapley_vote(probs, lt_idx.to(DEVICE), bill_idx.to(DEVICE), h["legislator_term"].size(0), h["bill"].size(0))
        topic_emb = h["topic"]
        actor_topic_rows: List[Dict[str, object]] = []
        actor_overall_rows: List[Dict[str, object]] = []
        for lt in range(h["legislator_term"].size(0)):
            topic_scores = []
            for t in range(topic_emb.size(0)):
                stance = self.model.stance_forward(h, "legislator_term", torch.tensor([lt], device=DEVICE), torch.tensor([t], device=DEVICE)).squeeze().item()
                engagement = torch.sigmoid((shapley[lt] * (topic_idx == t).float()).sum())
                influence_delta = (delta[bill_idx == bill_idx] * (topic_idx == t).to(delta.device)).mean().item() if (topic_idx == t).any() else 0.0
                actor_topic_rows.append(
                    {
                        "actor_id": lt,
                        "actor_type": "legislator_term",
                        "topic_id": t,
                        "stance": stance,
                        "stance_ci_lo": stance - 0.1,
                        "stance_ci_hi": stance + 0.1,
                        "influence_delta_mean": influence_delta,
                        "influence_ci_lo": influence_delta - 0.05,
                        "influence_ci_hi": influence_delta + 0.05,
                        "engagement": float(engagement),
                        "salience": float(torch.sigmoid(topic_emb[t].norm())),
                        "recency": 0.5,
                        "certainty": 0.8,
                        "topness_score": float(engagement),
                        "pathway_share": {"vote_share": 0.7, "committee_share": 0.3},
                        "support": {"n_votes": int((lt_idx == lt).sum().item()), "n_final": 0, "n_comm_reads": 0, "spend": 0.0, "lobby_touches": 0},
                        "evidence": {"top_paths": [], "pivotal_bills": []},
                    }
                )
                topic_scores.append((t, engagement))
            weights = torch.tensor([score for _, score in topic_scores])
            weights = weights / weights.sum().clamp_min(1e-6)
            influence_total = (weights * torch.tensor([row["influence_delta_mean"] for row in actor_topic_rows[-len(topic_scores):]])).sum().item()
            actor_overall_rows.append(
                {
                    "actor_id": lt,
                    "actor_type": "legislator_term",
                    "overall_influence": influence_total,
                    "ci_lo": influence_total - 0.1,
                    "ci_hi": influence_total + 0.1,
                    "topic_breakdown": [{"topic_id": t, "weight": float(w.item()), "delta": actor_topic_rows[-len(topic_scores) + i]["influence_delta_mean"]} for i, (t, w) in enumerate(zip([t for t, _ in topic_scores], weights))],
                    "top_topics": [t for t, _ in sorted(topic_scores, key=lambda x: x[1], reverse=True)[:5]],
                }
            )
        committee_rows = []
        for c in range(h["committee"].size(0)):
            committee_rows.append(
                {
                    "committee_id": c,
                    "overall_influence": 0.0,
                    "ci_lo": -0.05,
                    "ci_hi": 0.05,
                    "topic_breakdown": [],
                    "top_topics": [],
                    "gate_index": 0.0,
                }
            )
        return {"actor_topic": actor_topic_rows, "actor_overall": actor_overall_rows, "committee_overall": committee_rows}


def macro_f1(logits: Tensor, target: Tensor) -> Tensor:
    mask = target.ge(0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits = logits[mask]
    target = target[mask]
    pred = logits.argmax(dim=-1)
    f1 = 0.0
    for cls in range(logits.size(-1)):
        tp = ((pred == cls) & (target == cls)).sum().float()
        fp = ((pred == cls) & (target != cls)).sum().float()
        fn = ((pred != cls) & (target == cls)).sum().float()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 += 2 * precision * recall / (precision + recall + 1e-6)
    return f1 / logits.size(-1)


def build_outputs(model: LeGNN4, data: HeteroData, embeddings: Dict[str, Tensor], influence: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    bill = data.to(DEVICE)
    logits = model.outcome_forward(bill, embeddings, torch.zeros(data["bill"].num_nodes, device=DEVICE), data["bill"].topic_ix.to(DEVICE))
    probs = F.softmax(logits, dim=-1)
    per_bill = []
    for idx in range(probs.size(0)):
        per_bill.append(
            {
                "bill_id": idx,
                "P(pass)": float(probs[idx, 2].item()),
                "P(veto)": float(probs[idx, 1].item()),
                "P(fail)": float(probs[idx, 0].item()),
                "expected_margin": float((probs[idx, 2] - probs[idx, 0]).item()),
                "pivotal_actors": [],
                "committee_bottlenecks": [],
            }
        )
    return {"actor_topic": influence["actor_topic"], "actor_overall": influence["actor_overall"], "committee_overall": influence["committee_overall"], "per_bill": per_bill}


def run_training(data: HeteroData, epochs: int = 3, cfg: Optional[ModelConfig] = None):
    cfg = cfg or ModelConfig()
    trainer = Trainer(data, cfg)
    for epoch in range(epochs):
        stats = trainer.train_epoch()
        metrics = trainer.evaluate()
        print(f"epoch {epoch}: {stats} {metrics}")
    embeddings = trainer.embed_full()
    influence = trainer.compute_influence(embeddings)
    outputs = build_outputs(trainer.model, trainer.data, embeddings, influence)
    return trainer, embeddings, outputs


if __name__ == "__main__":
    graph = torch.load("data4.pt", weights_only=False)
    trainer, embeddings, outputs = run_training(graph, epochs=1)
