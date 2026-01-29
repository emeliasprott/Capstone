import os
import re
import json
import math
import time
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.transforms import (
    RemoveDuplicatedEdges,
    RemoveIsolatedNodes,
    ToUndirected,
)

# speed: allow faster matmul kernels (safe)
torch.set_float32_matmul_precision("high")


# ============================================================
# basics
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_default(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def nan_to_num_(t):
    if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def standardize_(t, eps=1e-6, clamp=8.0):
    if not (
        isinstance(t, torch.Tensor) and t.dtype.is_floating_point and t.numel() > 0
    ):
        return t
    mu = t.mean(dim=0, keepdim=True)
    sig = t.std(dim=0, keepdim=True, unbiased=False)
    sig = torch.where(sig < eps, torch.ones_like(sig), sig)
    x = (t - mu) / sig
    x = x.clamp(-clamp, clamp)
    t.copy_(x)
    return t


# ============================================================
# preprocess
# ============================================================
def normalize_vote_time(ea, cols=(0, 1)):
    t = ea[:, list(cols)].float()
    t = torch.nan_to_num(t)

    # normalize jointly, not per-column
    t = t - t.min()
    t = torch.log1p(t)

    mu = t.mean()
    sig = t.std(unbiased=False).clamp(min=1e-4)
    t = (t - mu) / sig
    t = t.clamp(-6, 6)

    ea = ea.clone()
    ea[:, list(cols)] = t
    return ea


def preprocess_data(data, standardize_x=True):
    for et in [
        ("legislator_term", "has_lobbying", "lobby_firm"),
        ("committee", "has_lobbying", "lobby_firm"),
        ("legislator_term", "has_donation", "donor"),
    ]:
        if et in data.edge_types:
            del data[et]

    data = ToUndirected()(data)
    data = RemoveDuplicatedEdges()(data)
    data = RemoveIsolatedNodes()(data)

    for nt in data.node_types:
        if "x" in data[nt] and data[nt].x is not None:
            x = data[nt].x
            if x.dtype != torch.float32:
                x = x.float()
            nan_to_num_(x)
            if standardize_x:
                standardize_(x)
            data[nt].x = x

    for et in data.edge_types:
        store = data[et]
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            ea = store.edge_attr
            if ea.dtype != torch.float32:
                ea = ea.float()
            nan_to_num_(ea)

            # precompute vote time normalization once
            if et == ("legislator_term", "voted_on", "bill_version") and ea.numel() > 0:
                ea = normalize_vote_time(ea, cols=(0, 1))

            store.edge_attr = ea

    return data


def money_intensity(amount, mu, sig):
    x = torch.clamp(amount, min=0.0)
    x = torch.log1p(x)
    x = (x - mu) / sig
    x = x.clamp(-6.0, 6.0)
    return F.softplus(x)


def compute_amount_stats(data, etype, amount_col=0):
    if etype not in data.edge_types:
        return None
    store = data[etype]
    if (
        not hasattr(store, "edge_attr")
        or store.edge_attr is None
        or store.edge_attr.numel() == 0
    ):
        return None
    ea = store.edge_attr
    if amount_col < 0 or amount_col >= ea.size(-1):
        return None
    x = ea[:, amount_col].detach().cpu().float()
    x = torch.clamp(x, min=0.0)
    x = torch.log1p(x)
    mu = float(x.mean().item())
    sig = float(x.std(unbiased=False).item())
    if (not np.isfinite(sig)) or sig < 1e-6:
        sig = 1.0
    return mu, sig


# ============================================================
# model components
# ============================================================
class MLP(nn.Module):
    def __init__(self, din, dh, dout, dropout=0.1, act="gelu", norm=True):
        super().__init__()
        self.l1 = nn.Linear(din, dh)
        self.l2 = nn.Linear(dh, dout)
        self.norm = nn.LayerNorm(dh) if norm else None
        self.dropout = float(dropout)
        self.act = act

    def forward(self, x):
        x = self.l1(x)
        if self.act == "gelu":
            x = F.gelu(x)
        else:
            x = F.relu(x)
        if self.norm is not None:
            x = self.norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.l2(x)
        return x


class PerTypeProjector(nn.Module):
    def __init__(self, in_dims, d_model=128, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleDict()
        for nt, din in in_dims.items():
            if din <= 0:
                continue
            self.proj[nt] = nn.Sequential(
                nn.Linear(din, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x_dict):
        out = {}
        for nt, x in x_dict.items():
            if nt in self.proj:
                out[nt] = self.proj[nt](x)
        return out


class EdgeGatedSAGELayer(nn.Module):
    def __init__(self, d_model, edge_dim=None, dropout=0.1):
        super().__init__()
        self.lin_src = nn.Linear(d_model, d_model, bias=False)
        self.lin_dst = nn.Linear(d_model, d_model)
        self.edge_gate = (
            MLP(edge_dim, d_model, 1, dropout=dropout, norm=False) if edge_dim else None
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x_src, x_dst, edge_index, edge_attr=None):
        src, dst = edge_index
        m = self.lin_src(x_src[src])

        if self.edge_gate is not None and edge_attr is not None:
            g = self.edge_gate(edge_attr).squeeze(-1)
            g = torch.sigmoid(g).unsqueeze(-1)
            m = m * g

        out = torch.zeros_like(x_dst)
        out.index_add_(0, dst, m)

        # fused shape: compute deg as (N, 1) directly (fewer ops)
        deg = torch.zeros((x_dst.size(0), 1), device=x_dst.device, dtype=torch.float)
        deg.index_add_(
            0,
            dst,
            torch.ones((dst.numel(), 1), device=x_dst.device, dtype=torch.float),
        )
        out = out / deg.clamp(min=1.0)

        out = self.norm(out + self.lin_dst(x_dst))
        return F.dropout(F.gelu(out), self.dropout, self.training)


class HeteroEncoder(nn.Module):
    def __init__(self, data, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = float(dropout)

        in_dims = {}
        for nt in data.node_types:
            if "x" in data[nt] and data[nt].x is not None:
                in_dims[nt] = int(data[nt].x.size(-1))
            else:
                in_dims[nt] = 0

        self.input_proj = PerTypeProjector(in_dims, d_model=d_model, dropout=dropout)

        # Per edge-type convs (fast; edge_attr only used where present)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            convs = nn.ModuleDict()
            for et in data.edge_types:
                store = data[et]
                edge_dim = None
                if (
                    hasattr(store, "edge_attr")
                    and store.edge_attr is not None
                    and store.edge_attr.numel() > 0
                ):
                    edge_dim = int(store.edge_attr.size(-1))
                key = self._ekey(et)
                convs[key] = EdgeGatedSAGELayer(
                    d_model, edge_dim=edge_dim, dropout=dropout
                )
            self.layers.append(convs)

        self.res_norms = nn.ModuleList(
            [
                nn.ModuleDict({nt: nn.LayerNorm(d_model) for nt in data.node_types})
                for _ in range(n_layers)
            ]
        )

    def _ekey(self, et):
        return f"{et[0]}__{et[1]}__{et[2]}"

    def forward(self, batch):
        x_dict = {
            nt: batch[nt].x
            for nt in batch.node_types
            if "x" in batch[nt] and batch[nt].x is not None
        }
        h = self.input_proj(x_dict)

        for nt in batch.node_types:
            if nt not in h:
                n = int(batch[nt].num_nodes)
                h[nt] = torch.zeros(
                    (n, self.d_model), device=next(self.parameters()).device
                )

        # allocator reuse: allocate buffers once per forward, then zero_ per layer
        h_new = {nt: torch.zeros_like(h[nt]) for nt in batch.node_types}

        for li, convs in enumerate(self.layers):
            for nt in batch.node_types:
                h_new[nt].zero_()

            for et in batch.edge_types:
                key = self._ekey(et)
                store = batch[et]
                ei = store.edge_index
                if ei is None or ei.numel() == 0 or ei.size(1) == 0:
                    continue

                src, _, dst = et
                ea = getattr(store, "edge_attr", None)
                if ea is not None and ea.numel() > 0 and ea.size(0) != ei.size(1):
                    ea = None

                msg = convs[key](h[src], h[dst], ei, ea)
                h_new[dst].add_(msg)

            for nt in batch.node_types:
                h_new[nt] = self.res_norms[li][nt](h_new[nt] + h[nt])

            h = h_new

            # re-init buffers (keep dict; recreate tensors only if shapes changed)
            h_new = {nt: torch.zeros_like(h[nt]) for nt in batch.node_types}

        return h


class ActorTopicHead(nn.Module):
    """
    Latent stance in [-1,1] and influence >= 0, conditioned on topic.
    """

    def __init__(self, d_model, num_topics, dropout=0.1):
        super().__init__()
        self.topic_emb = nn.Embedding(int(num_topics), int(d_model))
        self.stance_mlp = MLP(2 * d_model, 2 * d_model, 1, dropout=dropout, norm=True)
        self.infl_mlp = MLP(2 * d_model, 2 * d_model, 1, dropout=dropout, norm=True)

    def forward(self, h_actor, topic_ids):
        t = self.topic_emb(topic_ids)
        z = torch.cat([h_actor, t], dim=-1)
        stance = torch.tanh(self.stance_mlp(z).squeeze(-1))
        infl = F.softplus(self.infl_mlp(z).squeeze(-1))
        return stance, infl


class StanceInfluenceNet(nn.Module):
    def __init__(
        self,
        data,
        num_topics,
        d_model=128,
        n_layers=2,
        dropout=0.1,
        actor_types=("legislator_term", "committee", "donor", "lobby_firm"),
        vote_edge_dim=19,
    ):
        super().__init__()
        self.actor_types = list(actor_types)
        self.num_topics = int(num_topics)
        self.d_model = int(d_model)
        self.dropout = float(dropout)

        self.encoder = HeteroEncoder(
            data, d_model=d_model, n_layers=n_layers, dropout=dropout
        )

        self.heads = nn.ModuleDict(
            {
                at: ActorTopicHead(d_model, num_topics, dropout=dropout)
                for at in self.actor_types
            }
        )

        self.vote_ctx = MLP(vote_edge_dim, 2 * d_model, 1, dropout=dropout, norm=True)
        self.bias_bill = nn.Parameter(torch.zeros(()))

        self.rel_scale = nn.ParameterDict(
            {
                "vote": nn.Parameter(torch.tensor(1.0)),
                "wrote": nn.Parameter(torch.tensor(1.0)),
                "read": nn.Parameter(torch.tensor(1.0)),
                "donate": nn.Parameter(torch.tensor(1.0)),
                "lobby": nn.Parameter(torch.tensor(1.0)),
            }
        )


# routing helpers inside a sampled subgraph
def build_bv2bill(batch):
    et = ("bill_version", "is_version", "bill")
    if et not in batch.edge_types:
        return None
    ei = batch[et].edge_index
    if ei is None or ei.numel() == 0:
        return None
    n_bv = int(batch["bill_version"].num_nodes)
    out = torch.full((n_bv,), -1, device=ei.device, dtype=torch.long)
    out[ei[0]] = ei[1]
    return out


def vote_targets_from_edge_attr(ea, vote_target_col=-1):
    if ea is None or ea.numel() == 0:
        return None
    col = vote_target_col if vote_target_col >= 0 else (ea.size(-1) + vote_target_col)
    if col < 0 or col >= ea.size(-1):
        return None
    v = ea[:, col].float()
    v = torch.sign(v)
    keep = v != 0
    if keep.sum().item() == 0:
        return None
    y = (v[keep] > 0).float()
    return y, keep


def bill_topics_and_y(batch):
    if "bill" not in batch.node_types:
        return None
    if "cluster" not in batch["bill"] or "y" not in batch["bill"]:
        return None
    t = batch["bill"].cluster.long()
    y = batch["bill"].y.long()
    return t, y


# losses
def loss_vote_recon(model, batch, h, cfg):
    dev = next(model.parameters()).device
    et = ("legislator_term", "voted_on", "bill_version")

    if et not in batch.edge_types:
        return torch.tensor(0.0, device=dev)

    store = batch[et]
    ei = store.edge_index
    ea = getattr(store, "edge_attr", None)

    if ei is None or ei.numel() == 0:
        return torch.tensor(0.0, device=dev)

    # --- binary supervision: vote occurred ---
    y = torch.ones((ei.size(1),), device=dev)

    leg = ei[0]
    bv = ei[1]

    # --- bill -> topic routing (soft gate, not hard drop) ---
    bv2bill = build_bv2bill(batch)
    if bv2bill is None:
        return torch.tensor(0.0, device=dev)

    bill = bv2bill[bv]
    bill_topics, _ = bill_topics_and_y(batch)
    if bill_topics is None:
        return torch.tensor(0.0, device=dev)

    topics = bill_topics[bill]
    valid = topics >= 0
    if valid.sum() < cfg.min_votes_per_batch:
        return torch.tensor(0.0, device=dev)

    leg = leg[valid]
    bv = bv[valid]
    topics = topics[valid]
    y = y[valid]
    ea = ea[valid] if ea is not None else None

    # --- embeddings ---
    h_leg = h["legislator_term"][leg]
    h_bv = h["bill_version"][bv]

    # --- stance signal (latent, weak) ---
    s, p = model.heads["legislator_term"](h_leg, topics)
    base = (s * p) * model.rel_scale["vote"]

    # --- contextual signal ---
    ctx = 0.0
    if ea is not None and ea.numel() > 0:
        # ea already normalized in preprocess_data
        ctx = model.vote_ctx(ea).squeeze(-1)

    inter = (h_leg * h_bv).sum(dim=-1) / math.sqrt(model.d_model)

    logits = base + ctx + inter
    loss = F.binary_cross_entropy_with_logits(logits, y)

    return loss * float(cfg.vote_recon_weight)


def loss_version_smooth(model, batch, h, cfg):
    et = ("bill_version", "priorVersion", "bill_version")
    if et not in batch.edge_types or "bill_version" not in h:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    ei = batch[et].edge_index
    if ei is None or ei.numel() == 0 or ei.size(1) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    a = h["bill_version"][ei[0]]
    b = h["bill_version"][ei[1]]
    sim = F.cosine_similarity(a, b, dim=-1)
    return (1.0 - sim).mean() * float(cfg.temp_smooth_weight)


def loss_bill_outcome(model, batch, h, cfg, money_stats):
    """
    Build bill logits from in-batch routed actor contributions:
      - legislator_term via voted_on and wrote edges to bv -> bill
      - committee via read edges to bv -> bill
      - donor via donated_to -> legislator_term -> bill (in-batch)
      - lobby_firm via lobbied -> legislator_term/committee -> bill (in-batch)
    """
    ty = bill_topics_and_y(batch)
    if ty is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    bill_topics, y_full = ty

    eligible = (bill_topics >= 0) & (y_full >= 0)
    if eligible.sum().item() == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    bv2bill = build_bv2bill(batch)
    if bv2bill is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    dev = next(model.parameters()).device
    n_bill = int(batch["bill"].num_nodes)

    logits = torch.zeros((n_bill,), device=dev) + model.bias_bill

    # -------- helper: add actor->bill contributions from edges (actor -> bv) + bv2bill
    def add_from_actor_to_bv(
        actor_type, rel_name, rel_scale_key, weight_from_edge_attr=None
    ):
        et = (actor_type, rel_name, "bill_version")
        if et not in batch.edge_types or actor_type not in h:
            return

        ei = batch[et].edge_index
        a = ei[0]
        bv = ei[1]
        b = bv2bill[bv]
        m = b >= 0
        if m.sum().item() == 0:
            return
        a = a[m]
        b = b[m]

        t = bill_topics[b]
        m2 = t >= 0
        if m2.sum().item() == 0:
            return
        a = a[m2]
        b = b[m2]
        t = t[m2]

        h_a = h[actor_type][a]
        s, p = model.heads[actor_type](h_a, t)
        impact = (s * p) * model.rel_scale[rel_scale_key]

        w = torch.ones_like(impact)
        if weight_from_edge_attr is not None:
            store = batch[et]
            ea = getattr(store, "edge_attr", None)
            if ea is not None and ea.numel() > 0 and ea.size(0) == ei.size(1):
                ea = ea[m][m2]
                w = weight_from_edge_attr(ea, impact.device)

        logits.scatter_add_(0, b, impact * w)

    # -------- helper: add committee contributions from bv -> committee edges, routed back to bill
    def add_from_bv_to_actor(
        actor_type, rel_name, rel_scale_key, weight_from_edge_attr=None
    ):
        et = ("bill_version", rel_name, actor_type)
        if et not in batch.edge_types or actor_type not in h:
            return

        ei = batch[et].edge_index
        bv = ei[0]
        a = ei[1]
        b = bv2bill[bv]
        m = b >= 0
        if m.sum().item() == 0:
            return
        a = a[m]
        b = b[m]

        t = bill_topics[b]
        m2 = t >= 0
        if m2.sum().item() == 0:
            return
        a = a[m2]
        b = b[m2]
        t = t[m2]

        h_a = h[actor_type][a]
        s, p = model.heads[actor_type](h_a, t)
        impact = (s * p) * model.rel_scale[rel_scale_key]

        w = torch.ones_like(impact)
        if weight_from_edge_attr is not None:
            store = batch[et]
            ea = getattr(store, "edge_attr", None)
            if ea is not None and ea.numel() > 0 and ea.size(0) == ei.size(1):
                ea = ea[m][m2]
                w = weight_from_edge_attr(ea, impact.device)

        logits.scatter_add_(0, b, impact * w)

    # weights for authorship edges (optional)
    def author_weight(ea, dev):
        a = ea[:, 0].float()
        w = (0.6 + (a / 3.0).clamp(0.0, 1.5)).to(dev)
        return w

    # ===== direct bill routing =====
    add_from_actor_to_bv(
        "legislator_term", "voted_on", "vote", weight_from_edge_attr=None
    )
    add_from_actor_to_bv(
        "legislator_term", "wrote", "wrote", weight_from_edge_attr=author_weight
    )
    add_from_bv_to_actor("committee", "read", "read", weight_from_edge_attr=None)

    # supervised bill outcome loss on eligible bills in this batch
    y_bin = (y_full == 1).float()
    loss = F.binary_cross_entropy_with_logits(
        logits[eligible], y_bin[eligible].to(logits.device)
    )
    return loss * float(cfg.bill_outcome_weight)


def loss_regularize(model, h, cfg):
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    t = torch.randint(0, model.num_topics, (1,), device=reg.device).long()
    for at in model.actor_types:
        if at not in h:
            continue
        ha = h[at]
        if ha.numel() == 0:
            continue
        t_ids = t.expand(ha.size(0))
        s, p = model.heads[at](ha, t_ids)
        reg = reg + float(cfg.stance_l2) * (s.pow(2).mean())
        reg = reg + float(cfg.influence_l1) * (p.mean())
    return reg


# loaders (no train/test split)
def make_vote_link_loader(data, cfg):
    et = ("legislator_term", "voted_on", "bill_version")
    ei = data[et].edge_index

    loader = LinkNeighborLoader(
        data,
        num_neighbors=list(cfg.num_neighbors_vote),
        edge_label_index=(et, ei),
        edge_label=torch.ones(ei.size(1)),
        batch_size=cfg.vote_edge_batch_size,
        shuffle=True,
        neg_sampling_ratio=cfg.vote_neg_ratio,
        replace=False,
        directed=True,
        persistent_workers=False,
    )
    return loader


def make_bill_loader(data, cfg):
    if "bill" not in data.node_types:
        raise RuntimeError("No bill nodes in data")
    if "cluster" not in data["bill"] or "y" not in data["bill"]:
        raise RuntimeError("bill.cluster and bill.y required")

    eligible = (data["bill"].cluster >= 0) & (data["bill"].y >= 0)
    idx = torch.nonzero(eligible, as_tuple=False).squeeze(-1)

    loader = NeighborLoader(
        data,
        input_nodes=("bill", idx),
        num_neighbors=list(cfg.num_neighbors_bill),
        batch_size=int(cfg.bill_batch_size),
        shuffle=True,
        persistent_workers=False,
    )
    return loader


# ============================================================
# export
# ============================================================
@torch.no_grad()
def export_actor_topic_tables(model, data, cfg, out_path, device):
    model.eval()
    rows = []

    for at in cfg.actor_types:
        if at not in data.node_types:
            continue
        n = int(data[at].num_nodes)
        if n == 0:
            continue

        loader = NeighborLoader(
            data,
            input_nodes=(at, torch.arange(n)),
            num_neighbors=list(cfg.num_neighbors_actor),
            batch_size=int(cfg.actor_batch_size),
            shuffle=False,
            persistent_workers=False,
        )

        for batch in tqdm(loader, desc=f"export {at}", total=len(loader)):
            batch = batch.to(device)
            h = model.encoder(batch)
            bs = int(batch[at].batch_size)
            h_at = h[at][:bs]
            global_ids = batch[at].n_id[:bs].detach().cpu().numpy().astype(np.int64)

            # compute stance & influence for all topics, vectorized
            T = int(model.num_topics)
            topic_ids = torch.arange(T, device=h_at.device).long()
            # expand: [bs, T, d]
            h_exp = h_at.unsqueeze(1).expand(-1, T, -1)
            t_exp = model.heads[at].topic_emb(topic_ids).unsqueeze(0).expand(bs, -1, -1)
            z = torch.cat([h_exp, t_exp], dim=-1)

            s = torch.tanh(model.heads[at].stance_mlp(z).squeeze(-1))  # [bs, T]
            p = F.softplus(model.heads[at].infl_mlp(z).squeeze(-1))  # [bs, T]
            impact = s * p

            df = pd.DataFrame(
                {
                    "actor_type": at,
                    "actor_index": np.repeat(global_ids, T),
                    "topic_id": np.tile(np.arange(T, dtype=np.int64), bs),
                    "stance": s.detach().cpu().reshape(-1).numpy(),
                    "influence": p.detach().cpu().reshape(-1).numpy(),
                    "impact": impact.detach().cpu().reshape(-1).numpy(),
                }
            )
            rows.append(df)

    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "actor_type",
                "actor_index",
                "topic_id",
                "stance",
                "influence",
                "impact",
            ]
        )
    )
    out.to_parquet(out_path, index=False)


# ============================================================
# train
# ============================================================
def train(cfg):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    ckpt_dir = ensure_dir(os.path.join(cfg.out_dir, "checkpoints"))
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")

    data = torch.load(cfg.data_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise RuntimeError("data_path did not load HeteroData")

    data = preprocess_data(data, standardize_x=cfg.standardize_x)

    # topic count from bill.cluster
    cl = data["bill"].cluster
    num_topics = int(cl[cl >= 0].max().item() + 1)

    # money stats (log1p standardized) for edges that exist
    money_stats = {}
    for et in [
        ("donor", "donated_to", "legislator_term"),
        ("lobby_firm", "lobbied", "legislator_term"),
        ("lobby_firm", "lobbied", "committee"),
    ]:
        st = compute_amount_stats(data, et, amount_col=cfg.money_amount_col)
        if st is not None:
            money_stats[str(et)] = st

    # infer vote edge dim
    vote_et = ("legislator_term", "voted_on", "bill_version")
    vote_edge_dim = (
        int(data[vote_et].edge_attr.size(-1))
        if hasattr(data[vote_et], "edge_attr") and data[vote_et].edge_attr is not None
        else 0
    )

    model = StanceInfluenceNet(
        data=data,
        num_topics=num_topics,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        actor_types=tuple(cfg.actor_types),
        vote_edge_dim=vote_edge_dim if vote_edge_dim > 0 else 19,
    )

    dev = (
        device_default(cfg.device_preference)
        if cfg.device is None
        else torch.device(cfg.device)
    )
    model = model.to(dev)
    if cfg.export_only == True:
        model_dict = torch.load(os.path.join(ckpt_dir, "epoch_010.pt"))
        model.load_state_dict(model_dict["model"])
        out_path = os.path.join(cfg.out_dir, cfg.export_name)
        export_actor_topic_tables(model, data, cfg, out_path, dev)
        print(f"Saved actor-topic parquet: {out_path}")
        return

    vote_loader = make_vote_link_loader(data, cfg)
    bill_loader = make_bill_loader(data, cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # iterators
    vote_iter = iter(vote_loader)
    bill_iter = iter(bill_loader)

    def next_batch(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    with open(metrics_path, "a", encoding="utf-8") as mf:
        for epoch in range(cfg.epochs):
            model.train()
            losses = {"vote": [], "bill": [], "temp": [], "reg": [], "total": []}

            pbar = tqdm(
                range(cfg.steps_per_epoch),
                desc=f"train e{epoch}",
                total=cfg.steps_per_epoch,
            )
            for step in pbar:
                opt.zero_grad(set_to_none=True)

                total = torch.tensor(0.0, device=dev)

                # stop double encoder passes: if a bill step is scheduled, do bill only
                do_bill = cfg.bill_outcome_weight > 0 and (step % cfg.bill_every == 0)
                do_vote = (
                    cfg.vote_recon_weight > 0
                    and (step % cfg.vote_every == 0)
                    and (not do_bill)
                )

                # --- bill step (anchors geometry to outcomes, topic-conditioned) ---
                if do_bill:
                    bb, bill_iter = next_batch(bill_iter, bill_loader)
                    bb = bb.to(dev)
                    h = model.encoder(bb)
                    lb = loss_bill_outcome(model, bb, h, cfg, money_stats)
                    lt2 = (
                        loss_version_smooth(model, bb, h, cfg)
                        if cfg.temp_smooth_weight > 0
                        else torch.tensor(0.0, device=dev)
                    )
                    lr2 = (
                        loss_regularize(model, h, cfg)
                        if (cfg.stance_l2 > 0 or cfg.influence_l1 > 0)
                        else torch.tensor(0.0, device=dev)
                    )
                    total = total + lb + lt2 + lr2
                    losses["bill"].append(float(lb.detach().cpu().item()))
                    if cfg.temp_smooth_weight > 0:
                        losses["temp"].append(float(lt2.detach().cpu().item()))
                    losses["reg"].append(float(lr2.detach().cpu().item()))

                # --- vote step (dominant signal) ---
                if do_vote:
                    vb, vote_iter = next_batch(vote_iter, vote_loader)
                    vb = vb.to(dev)
                    h = model.encoder(vb)
                    lv = loss_vote_recon(model, vb, h, cfg)
                    lt = (
                        loss_version_smooth(model, vb, h, cfg)
                        if cfg.temp_smooth_weight > 0
                        else torch.tensor(0.0, device=dev)
                    )
                    lr = (
                        loss_regularize(model, h, cfg)
                        if (cfg.stance_l2 > 0 or cfg.influence_l1 > 0)
                        else torch.tensor(0.0, device=dev)
                    )
                    total = total + lv + lt + lr
                    losses["vote"].append(float(lv.detach().cpu().item()))
                    losses["temp"].append(float(lt.detach().cpu().item()))
                    losses["reg"].append(float(lr.detach().cpu().item()))

                if not torch.isfinite(total):
                    continue

                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                losses["total"].append(float(total.detach().cpu().item()))
                if len(losses["total"]) > 0:
                    pbar.set_postfix(
                        {
                            "tot": np.mean(losses["total"][-50:]),
                            "vote": (
                                np.mean(losses["vote"][-50:]) if losses["vote"] else 0.0
                            ),
                            "bill": (
                                np.mean(losses["bill"][-50:]) if losses["bill"] else 0.0
                            ),
                        }
                    )

            rec = {
                "ts": time.time(),
                "epoch": epoch,
                "loss_total": (
                    float(np.mean(losses["total"])) if losses["total"] else None
                ),
                "loss_vote": float(np.mean(losses["vote"])) if losses["vote"] else None,
                "loss_bill": float(np.mean(losses["bill"])) if losses["bill"] else None,
                "loss_temp": float(np.mean(losses["temp"])) if losses["temp"] else None,
                "loss_reg": float(np.mean(losses["reg"])) if losses["reg"] else None,
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()

            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": {
                        k: getattr(cfg, k)
                        for k in cfg.__dict__.keys()
                        if not k.startswith("_")
                    },
                    "num_topics": num_topics,
                    "money_stats": money_stats,
                },
                ckpt_path,
            )

            print(
                f"[epoch {epoch}] total={rec['loss_total']:.5f} vote={rec['loss_vote']} bill={rec['loss_bill']}"
            )

    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                k: getattr(cfg, k) for k in cfg.__dict__.keys() if not k.startswith("_")
            },
            "num_topics": num_topics,
            "money_stats": money_stats,
        },
        final_path,
    )
    print(f"Saved final checkpoint: {final_path}")

    if cfg.export_after_train:
        out_path = os.path.join(cfg.out_dir, cfg.export_name)
        export_actor_topic_tables(model, data, cfg, out_path, dev)
        print(f"Saved actor-topic parquet: {out_path}")


# ============================================================
# config
# ============================================================
@dataclass
class Config:
    data_path: str = "data5.pt"
    out_dir: str = "dashboard/backend/data/outs"

    seed: int = 13
    device_preference: str = "mps"
    device: str = None

    # model
    d_model: int = 128
    n_layers: int = 2
    dropout: float = 0.12
    actor_types: list = field(
        default_factory=lambda: ["legislator_term", "committee", "donor", "lobby_firm"]
    )

    # optimization
    epochs: int = 10
    lr: float = 2e-4
    wd: float = 1e-3
    grad_clip: float = 1.0
    steps_per_epoch: int = 24

    # sampling
    min_votes_per_batch = 8
    vote_edge_batch_size: int = 7500
    bill_batch_size: int = 1200
    actor_batch_size: int = 512

    num_neighbors_vote: list = field(default_factory=lambda: [14, 7])
    num_neighbors_bill: list = field(default_factory=lambda: [12, 8])
    num_neighbors_actor: list = field(default_factory=lambda: [20, 12])

    # schedule (no split)
    vote_every: int = 1
    bill_every: int = 4
    vote_neg_ratio: float = 0.5

    # losses
    vote_recon_weight: float = 1.0
    bill_outcome_weight: float = 0.35
    temp_smooth_weight: float = 0.10
    stance_l2: float = 1e-3
    influence_l1: float = 1e-4

    # vote edge_attr interpretation
    vote_target_col: int = -1
    vote_time_col: int = None  # set to int column index if a raw time column exists

    # money
    use_money_routing: bool = True
    money_amount_col: int = 0

    # outputs
    export_after_train: bool = True
    export_only = True
    export_name: str = "actor_topic.parquet"

    # stability
    standardize_x: bool = True


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    train(cfg)


if __name__ == "__main__":
    main()
