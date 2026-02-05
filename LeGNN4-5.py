import os
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_default(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def sanitize_(x):
    x = x.float()
    torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def standardize_(x, eps=1e-6, clamp=8.0):
    mu = x.mean(0, keepdim=True)
    sig = x.std(0, keepdim=True, unbiased=False).clamp(min=eps)
    x.sub_(mu).div_(sig).clamp_(-clamp, clamp)
    return x


def inflate_to_2024_dollars(amount, year):
    inflation_factors = {
        2000: 0.5489093272878838,
        2001: 0.5643700652485646,
        2002: 0.5733755466380334,
        2003: 0.5865517081990259,
        2004: 0.6021984000527046,
        2005: 0.6224673743894734,
        2006: 0.6425238299913876,
        2007: 0.6609677996925917,
        2008: 0.6861834132315227,
        2009: 0.6839851725678692,
        2010: 0.6951790656295045,
        2011: 0.7170052709959213,
        2012: 0.7318701595537531,
        2013: 0.7425991678828936,
        2014: 0.754595584817024,
        2015: 0.7555096810253179,
        2016: 0.7650847126241973,
        2017: 0.7813920720948556,
        2018: 0.8004502209397898,
        2019: 0.8149644535851114,
        2020: 0.8251754474980435,
        2021: 0.8637863825432861,
        2022: 0.9328257500450274,
        2023: 0.971330160074424,
        2024: 1.0,
        2025: 1.0264081092898847,
    }

    if isinstance(year, torch.Tensor):
        year_np = year.cpu().numpy()
        factors = np.vectorize(lambda y: inflation_factors.get(int(y), 1.0))(year_np)
        return amount * torch.tensor(factors, device=amount.device, dtype=amount.dtype)
    else:
        factor = inflation_factors.get(int(year), 1.0)
        return amount * factor


def get_bill_topic_tensor(data):
    return data["bill"]["cluster"].long()


def get_bill_outcome_tensor(data):
    return data["bill"]["y"].long()


def get_bill_term_tensor(data):
    return data["bill"].x[:, 1].long()


def build_term_index(term_tensor):
    vals = term_tensor.detach().cpu().numpy().astype(np.int64)
    uniq = np.unique(vals)
    uniq = uniq[np.isfinite(uniq)]
    uniq = np.sort(uniq)
    term_to_idx = {int(v): i for i, v in enumerate(uniq.tolist())}
    idx = torch.tensor([term_to_idx[int(v)] for v in vals], dtype=torch.long)
    return idx, term_to_idx, uniq


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


MONETARY_EDGES = [
    ("donor", "expenditure", "bill"),
    ("donor", "donated_to", "legislator_term"),
    ("lobby_firm", "lobbied", "legislator_term"),
    ("lobby_firm", "lobbied", "committee"),
]


def preprocess_data(data, standardize_x=True, keep_bidir=True):
    data = data.clone()
    if keep_bidir:
        data = ToUndirected()(data)

    data = RemoveDuplicatedEdges()(data)
    data = RemoveIsolatedNodes()(data)

    for nt in data.node_types:
        if "x" in data[nt]:
            x = sanitize_(data[nt].x)
            if standardize_x:
                standardize_(x)
            data[nt].x = x

    for et in data.edge_types:
        store = data[et]
        if "edge_attr" in store:
            ea = sanitize_(store.edge_attr)

            if et in MONETARY_EDGES:
                amount = ea[:, 0]
                year = ea[:, 2]

                ea[:, 0] = inflate_to_2024_dollars(amount, year)

            store.edge_attr = ea

    return data


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
        x = F.gelu(x) if self.act == "gelu" else F.relu(x)
        if self.norm is not None:
            x = self.norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.l2(x)


class PerTypeProjector(nn.Module):
    def __init__(self, in_dims, d_model=128, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleDict()
        for nt, din in in_dims.items():
            if din > 0:
                self.proj[nt] = nn.Sequential(
                    nn.Linear(din, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                )

    def forward(self, x_dict):
        return {nt: self.proj[nt](x) for nt, x in x_dict.items() if nt in self.proj}


class EdgeGatedSAGELayer(nn.Module):
    def __init__(self, d_model, edge_dim=None, dropout=0.1):
        super().__init__()
        self.src = nn.Linear(d_model, d_model, bias=False)
        self.dst = nn.Linear(d_model, d_model)
        self.gate = MLP(edge_dim, d_model, 1, norm=False) if edge_dim else None
        self.norm = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x_src, x_dst, edge_index, edge_attr=None):
        s, d = edge_index
        msg = self.src(x_src[s])
        if self.gate is not None and edge_attr is not None:
            msg = msg * torch.sigmoid(self.gate(edge_attr))

        out = torch.zeros_like(x_dst)
        out.index_add_(0, d, msg)

        deg = torch.bincount(d, minlength=x_dst.size(0)).clamp(min=1).unsqueeze(-1)
        out = out / deg
        out = self.norm(out + self.dst(x_dst))
        return F.gelu(out)


class HeteroEncoder(nn.Module):
    def __init__(self, data, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.input_proj = PerTypeProjector(
            {
                nt: data[nt].x.size(-1) if "x" in data[nt] else 0
                for nt in data.node_types
            },
            d_model=d_model,
            dropout=dropout,
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        f"{s}__{r}__{d}": EdgeGatedSAGELayer(
                            d_model,
                            (
                                data[(s, r, d)].edge_attr.size(-1)
                                if "edge_attr" in data[(s, r, d)]
                                else None
                            ),
                            dropout,
                        )
                        for (s, r, d) in data.edge_types
                    }
                )
            )

        self.norms = nn.ModuleDict(
            {nt: nn.LayerNorm(d_model) for nt in data.node_types}
        )

    def forward(self, batch, active_edge_types=None):
        if active_edge_types is None:
            active_edge_types = batch.edge_types

        h = self.input_proj(
            {nt: batch[nt].x for nt in batch.node_types if "x" in batch[nt]}
        )

        for nt in batch.node_types:
            if nt not in h:
                h[nt] = torch.zeros(
                    batch[nt].num_nodes, self.d_model, device=batch[nt].x.device
                )

        for convs in self.layers:
            out = {nt: h[nt] for nt in h}
            for s, r, d in active_edge_types:
                key = f"{s}__{r}__{d}"
                store = batch[(s, r, d)]
                edge_attr = getattr(store, "edge_attr", None)
                out[d] = out[d] + convs[key](h[s], h[d], store.edge_index, edge_attr)
            h = {nt: self.norms[nt](out[nt]) for nt in out}

        return h


class ActorTopicFactorHead(nn.Module):

    def __init__(self, d_model, num_topics, num_terms, rank=24, dropout=0.1):
        super().__init__()
        self.num_topics = int(num_topics)
        self.num_terms = int(num_terms)
        self.rank = int(rank)

        # Shared topic factors
        self.topic_factor = nn.Embedding(self.num_topics, self.rank)
        nn.init.normal_(self.topic_factor.weight, std=0.02)

        # Term-specific biases
        self.term_bias = nn.Embedding(self.num_terms, 1)
        nn.init.zeros_(self.term_bias.weight)

        # Actor-specific projections
        self.actor_to_p = MLP(
            d_model, 2 * d_model, self.rank, dropout=dropout, norm=True
        )
        self.actor_to_q = MLP(
            d_model, 2 * d_model, self.rank, dropout=dropout, norm=True
        )

        # Topic-specific biases
        self.topic_bias = nn.Embedding(self.num_topics, 1)
        nn.init.zeros_(self.topic_bias.weight)

        # Learned temperature parameters (not clamped during forward, only regularized)
        self.infl_scale = nn.Parameter(torch.tensor(1.0))
        self.stance_scale = nn.Parameter(torch.tensor(1.0))

    def stance(self, h_actor, topic_ids, term_ids):
        """Compute stance: positive = support, negative = oppose."""
        r = self.topic_factor(topic_ids)  # [B, R]
        p = self.actor_to_p(h_actor)  # [B, R]
        base = (p * r).sum(dim=-1)  # [B]

        tb = self.topic_bias(topic_ids).squeeze(-1)
        tr = self.term_bias(term_ids).squeeze(-1)

        # Scale controls sharpness (higher = sharper distinctions)
        scale = F.softplus(self.stance_scale) + 0.1
        return (base + tb + tr) / scale

    def influence(self, h_actor, topic_ids, term_ids):
        """Compute influence: always positive, measures strength of effect."""
        r = self.topic_factor(topic_ids)
        q = self.actor_to_q(h_actor)
        base = (q * r).sum(dim=-1)

        tr = self.term_bias(term_ids).squeeze(-1)
        scale = F.softplus(self.infl_scale) + 0.1
        return F.softplus(base / scale + tr)

    def expected_stance(self, h_actor, topic_probs, term_ids):
        """Expected stance over topic distribution."""
        p = self.actor_to_p(h_actor)  # [B, R]
        r = self.topic_factor.weight  # [T, R]
        z = torch.einsum("br,tr->bt", p, r)  # [B, T]
        z = z + self.topic_bias.weight.T
        z = z + self.term_bias(term_ids)
        scale = F.softplus(self.stance_scale) + 0.1
        z = z / scale
        eta = (topic_probs * z).sum(dim=-1)
        return eta, z

    def expected_influence(self, h_actor, topic_probs, term_ids):
        """Expected influence over topic distribution."""
        q = self.actor_to_q(h_actor)
        r = self.topic_factor.weight
        base = torch.einsum("br,tr->bt", q, r)
        base = base + self.term_bias(term_ids)
        scale = F.softplus(self.infl_scale) + 0.1
        infl = F.softplus(base / scale)
        eta = (topic_probs * infl).sum(dim=-1)
        return eta, infl


class BillTopicMixture(nn.Module):
    def __init__(self, d_model, num_topics, dropout=0.1):
        super().__init__()
        self.num_topics = int(num_topics)
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, self.num_topics),
        )

    def forward(self, h_bill):
        return self.net(h_bill)


class LegislativeStanceModel(nn.Module):
    def __init__(
        self,
        data,
        num_topics,
        num_terms,
        d_model=128,
        n_layers=2,
        dropout=0.1,
        rank=24,
        use_soft_topics=True,
        vote_edge_dim=None,
        exp_edge_dim=None,
        donate_edge_dim=None,
        lobby_edge_dim=None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_topics = int(num_topics)
        self.num_terms = int(num_terms)
        self.use_soft_topics = bool(use_soft_topics)

        self.encoder = HeteroEncoder(
            data, d_model=d_model, n_layers=n_layers, dropout=dropout
        )

        self.head_leg = ActorTopicFactorHead(
            d_model, num_topics, num_terms, rank=rank, dropout=dropout
        )
        self.head_donor = ActorTopicFactorHead(
            d_model, num_topics, num_terms, rank=rank, dropout=dropout
        )
        self.head_lobby = ActorTopicFactorHead(
            d_model, num_topics, num_terms, rank=rank, dropout=dropout
        )

        self.bill_topic_mix = (
            BillTopicMixture(d_model, num_topics, dropout=dropout)
            if self.use_soft_topics
            else None
        )

        ve = int(vote_edge_dim or 0)
        ee = int(exp_edge_dim or 0)
        de = int(donate_edge_dim or 0)
        le = int(lobby_edge_dim or 0)

        self.vote_ctx = (
            MLP(ve, 2 * d_model, 3, dropout=dropout, norm=True) if ve > 0 else None
        )
        self.exp_ctx = (
            MLP(ee, 2 * d_model, 2, dropout=dropout, norm=True) if ee > 0 else None
        )
        self.donate_ctx = (
            MLP(de, 2 * d_model, 1, dropout=dropout, norm=True) if de > 0 else None
        )
        self.lobby_ctx = (
            MLP(le, 2 * d_model, 1, dropout=dropout, norm=True) if le > 0 else None
        )

        # Bill latent variables
        self.bill_lean = nn.Linear(d_model, 1)  # Ideological position
        self.bill_sal = nn.Linear(d_model, 1)  # Salience/importance

        # Relation-specific scales
        self.rel_scale = nn.ParameterDict(
            {
                "vote": nn.Parameter(torch.tensor(1.0)),
                "exp": nn.Parameter(torch.tensor(1.0)),
                "donate": nn.Parameter(torch.tensor(1.0)),
                "lobby": nn.Parameter(torch.tensor(1.0)),
            }
        )

        self.abstain_bias = nn.Parameter(torch.tensor(0.0))
        self.out_bias = nn.Parameter(torch.tensor(0.0))
        self.amount_sigma = nn.Parameter(torch.tensor(0.6))

    def bill_latents(self, h_bill):
        """Extract bill ideological lean and salience."""
        lam = self.bill_lean(h_bill).squeeze(-1)
        kap = F.softplus(self.bill_sal(h_bill).squeeze(-1))
        return lam, kap

    def _topic_probs(self, h_bill):
        if not self.use_soft_topics:
            return None
        logits = self.bill_topic_mix(h_bill)
        return F.softmax(logits, dim=-1)


def make_vote_loader(data, cfg):
    et = ("legislator_term", "voted_on", "bill_version")
    store = data[et]
    ei = store.edge_index
    ea = getattr(store, "edge_attr", None)

    if ea is None or ea.numel() == 0:
        raise RuntimeError("vote edge_attr missing")

    v = ea[:, cfg.vote_target_col].float().clamp(-1.0, 1.0)
    y = torch.full((v.size(0),), 1, dtype=torch.long)
    y[v < 0] = 0
    y[v == 0] = 1
    y[v > 0] = 2

    return LinkNeighborLoader(
        data,
        edge_label_index=(et, ei),
        edge_label=y,
        num_neighbors=list(cfg.num_neighbors_vote),
        batch_size=int(cfg.vote_edge_batch_size),
        shuffle=True,
        neg_sampling_ratio=0.0,
        replace=False,
        directed=True,
        persistent_workers=False,
    )


def make_expenditure_loader(data, cfg):
    et = ("donor", "expenditure", "bill")
    if et not in data.edge_types:
        return None

    store = data[et]
    ei = store.edge_index
    ea = getattr(store, "edge_attr", None)

    if ea is None or ea.numel() == 0:
        raise RuntimeError("expenditure edge_attr missing")

    s = ea[:, cfg.exp_stance_col].float()
    y = (s > 0).long()

    return LinkNeighborLoader(
        data,
        edge_label_index=(et, ei),
        edge_label=y,
        num_neighbors=list(cfg.num_neighbors_exp),
        batch_size=int(cfg.exp_edge_batch_size),
        shuffle=True,
        neg_sampling_ratio=0.0,
        replace=False,
        directed=True,
        persistent_workers=False,
    )


def make_bill_loader(data, cfg):
    idx = torch.arange(int(data["bill"].num_nodes))
    return NeighborLoader(
        data,
        input_nodes=("bill", idx),
        num_neighbors=list(cfg.num_neighbors_bill),
        batch_size=int(cfg.bill_batch_size),
        shuffle=True,
        persistent_workers=False,
    )


def label_smooth_ce(logits, target, eps=0.05):
    """Cross-entropy with label smoothing."""
    n = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true = torch.zeros_like(logits).scatter_(1, target.view(-1, 1), 1.0)
        true = true * (1.0 - eps) + eps / float(n)
    return -(true * logp).sum(dim=-1).mean()


def vote_direction_loss(model, batch, h, bill_term_idx_full, cfg):
    """Loss for vote direction: -1 (no), 0 (abstain), +1 (yes)."""
    et = ("legislator_term", "voted_on", "bill_version")
    device = next(model.parameters()).device

    if et not in batch.edge_types:
        return torch.tensor(0.0, device=device)

    store = batch[et]
    ei = store.edge_index
    if ei is None or ei.numel() == 0:
        return torch.tensor(0.0, device=device)

    y = store.edge_attr[:, -1]
    leg = ei[0]
    bv = ei[1]

    bv2bill = build_bv2bill(batch)
    if bv2bill is None:
        return torch.tensor(0.0, device=device)

    bill = bv2bill[bv]
    m = bill >= 0
    if m.sum().item() == 0:
        return torch.tensor(0.0, device=device)

    leg, bv, bill, y = leg[m], bv[m], bill[m], y[m].long()

    h_leg = h["legislator_term"][leg]
    h_bv = h["bill_version"][bv]
    h_bill = h.get("bill")
    h_bill = h_bill[bill] if h_bill is not None else None

    topic_hard = get_bill_topic_tensor(batch)[bill]
    term_global = batch["bill"].n_id[bill] if hasattr(batch["bill"], "n_id") else None
    term_ids = (
        bill_term_idx_full[term_global.cpu()].to(device)
        if term_global is not None
        else torch.zeros_like(topic_hard)
    )

    # Compute stance
    topic_probs = (
        model._topic_probs(h_bill)
        if model.use_soft_topics and h_bill is not None
        else None
    )
    if topic_probs is None:
        z = model.head_leg.stance(h_leg, topic_hard, term_ids)
    else:
        z, _ = model.head_leg.expected_stance(h_leg, topic_probs, term_ids)

    # Bill lean
    lam = 0.0
    if h_bill is not None:
        lam, _ = model.bill_latents(h_bill)

    # Interaction term
    inter = (h_leg * h_bv).sum(dim=-1) / math.sqrt(model.d_model)

    # Context from edge features
    ea = getattr(store, "edge_attr", None)
    ctx_logits = None
    if model.vote_ctx is not None and ea is not None and ea.size(0) == ei.size(1):
        ctx_logits = model.vote_ctx(ea[m])

    # Combine signals
    base = model.rel_scale["vote"] * (z + 0.5 * lam + inter)
    logit_no = -base
    logit_abs = model.abstain_bias + (
        ctx_logits[:, 1] if ctx_logits is not None else 0.0
    )
    logit_yes = base

    if ctx_logits is not None:
        logit_no = logit_no + ctx_logits[:, 0]
        logit_yes = logit_yes + ctx_logits[:, 2]

    logits = torch.stack([logit_no, logit_abs, logit_yes], dim=-1)
    loss = label_smooth_ce(logits, y, eps=float(cfg.vote_label_smooth))
    return loss * float(cfg.vote_dir_weight)


def exp_stance_and_amount_loss(model, batch, h, bill_term_idx_full, cfg):
    """Loss for expenditure stance and amount."""
    et = ("donor", "expenditure", "bill")
    device = next(model.parameters()).device

    if et not in batch.edge_types:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    store = batch[et]
    ei = store.edge_index
    if ei is None or ei.numel() == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    ea = getattr(store, "edge_attr", None)
    if ea is None or ea.numel() == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    y = ea[:, -1]
    donor = ei[0]
    bill = ei[1]

    h_d = h["donor"][donor]
    h_b = h["bill"][bill]

    topic_hard = get_bill_topic_tensor(batch)[bill]
    term_global = batch["bill"].n_id[bill] if hasattr(batch["bill"], "n_id") else None
    term_ids = (
        bill_term_idx_full[term_global.cpu()].to(device)
        if term_global is not None
        else torch.zeros_like(topic_hard)
    )

    # Stance
    topic_probs = model._topic_probs(h_b) if model.use_soft_topics else None
    if topic_probs is None:
        z = model.head_donor.stance(h_d, topic_hard, term_ids)
    else:
        z, _ = model.head_donor.expected_stance(h_d, topic_probs, term_ids)

    lam, kap = model.bill_latents(h_b)
    inter = (h_d * h_b).sum(dim=-1) / math.sqrt(model.d_model)

    ctx = 0.0
    amt_logits = 0.0
    if model.exp_ctx is not None:
        out = model.exp_ctx(ea)
        ctx = out[:, 0]
        amt_logits = out[:, 1]

    logit = model.rel_scale["exp"] * (z + 0.5 * lam + inter) + ctx
    loss_stance = F.binary_cross_entropy_with_logits(logit, y.float()) * float(
        cfg.exp_stance_weight
    )

    # Amount loss
    amt = ea[:, cfg.exp_amount_col].float().clamp(min=0.0)
    z_amt = torch.log1p(amt)

    phi = model.head_donor.influence(h_d, topic_hard, term_ids)
    mu = (
        float(cfg.amt_a0)
        + float(cfg.amt_a1) * kap
        + float(cfg.amt_a2) * phi
        + float(cfg.amt_a3) * torch.abs(z)
        + float(cfg.amt_a4) * torch.abs(logit.detach())
        + amt_logits
    )

    sig = F.softplus(model.amount_sigma).clamp(min=0.05, max=5.0)
    nll = 0.5 * ((z_amt - mu) / sig).pow(2) + torch.log(sig + 1e-8)
    loss_amt = nll.mean() * float(cfg.exp_amount_weight)

    return loss_stance, loss_amt


def bill_topic_mixture_loss(model, batch, h, cfg):
    """Loss for soft topic prediction on bills."""
    device = next(model.parameters()).device

    if not model.use_soft_topics or "bill" not in batch.node_types:
        return torch.tensor(0.0, device=device)

    h_b = h["bill"]
    logits = model.bill_topic_mix(h_b)

    t = get_bill_topic_tensor(batch)
    if t.numel() == 0:
        return torch.tensor(0.0, device=device)

    keep = (t >= 0) & (t < model.num_topics)
    if keep.sum().item() == 0:
        return torch.tensor(0.0, device=device)

    loss = label_smooth_ce(logits[keep], t[keep], eps=float(cfg.topic_label_smooth))
    return loss * float(cfg.topic_ce_weight)


def version_smooth_loss(model, batch, h, cfg):
    """Temporal smoothness: consecutive bill versions should have similar embeddings."""
    et = ("bill_version", "priorVersion", "bill_version")
    device = next(model.parameters()).device

    if et not in batch.edge_types or "bill_version" not in h:
        return torch.tensor(0.0, device=device)

    ei = batch[et].edge_index
    if ei is None or ei.numel() == 0:
        return torch.tensor(0.0, device=device)

    a = h["bill_version"][ei[0]]
    b = h["bill_version"][ei[1]]
    sim = F.cosine_similarity(a, b, dim=-1)
    return (1.0 - sim).mean() * float(cfg.version_smooth_weight)


def outcome_aux_loss(model, batch, h, bill_term_idx_full, cfg):
    """Auxiliary loss: predict bill outcome from aggregated actor signals."""
    device = next(model.parameters()).device

    if cfg.outcome_weight <= 0 or "bill" not in batch.node_types:
        return torch.tensor(0.0, device=device)

    y = get_bill_outcome_tensor(batch)
    keep = (y == 0) | (y == 1)
    if keep.sum().item() == 0:
        return torch.tensor(0.0, device=device)

    h_bill = h["bill"][keep]
    y = y[keep].float()
    bill_ids_local = torch.nonzero(keep, as_tuple=False).squeeze(-1)

    topic_hard = get_bill_topic_tensor(batch)[keep]
    term_global = (
        batch["bill"].n_id[bill_ids_local] if hasattr(batch["bill"], "n_id") else None
    )
    term_ids = (
        bill_term_idx_full[term_global.cpu()].to(device)
        if term_global is not None
        else torch.zeros_like(topic_hard)
    )

    # Base prediction from bill latents
    lam, kap = model.bill_latents(h_bill)
    base = model.out_bias + float(cfg.out_lam_w) * lam - float(cfg.out_kap_w) * kap
    logits = base.clone()

    bv2bill = build_bv2bill(batch)
    if bv2bill is None:
        return F.binary_cross_entropy_with_logits(logits, y) * float(cfg.outcome_weight)

    # Map bill IDs
    bill_map = torch.full(
        (int(batch["bill"].num_nodes),), -1, device=device, dtype=torch.long
    )
    bill_map[bill_ids_local] = torch.arange(bill_ids_local.numel(), device=device)

    # Aggregate legislator votes
    et_vote = ("legislator_term", "voted_on", "bill_version")
    if et_vote in batch.edge_types and "legislator_term" in h and "bill_version" in h:
        ei = batch[et_vote].edge_index
        if ei is not None and ei.numel() > 0:
            leg = ei[0]
            bv = ei[1]
            bill_full = bv2bill[bv]
            m = bill_full >= 0
            if m.sum().item() > 0:
                leg, bill_full = leg[m], bill_full[m]
                idx = bill_map[bill_full]
                m2 = idx >= 0
                if m2.sum().item() > 0:
                    leg, idx = leg[m2], idx[m2]
                    h_leg = h["legislator_term"][leg]
                    z = model.head_leg.stance(h_leg, topic_hard[idx], term_ids[idx])
                    logits.index_add_(0, idx, float(cfg.out_vote_w) * z)

    # Aggregate donor expenditures
    et_exp = ("donor", "expenditure", "bill")
    if et_exp in batch.edge_types and "donor" in h:
        ei = batch[et_exp].edge_index
        ea = getattr(batch[et_exp], "edge_attr", None)
        if ei is not None and ei.numel() > 0:
            d, b = ei[0], ei[1]
            idx = bill_map[b]
            m = idx >= 0
            if m.sum().item() > 0:
                d, idx = d[m], idx[m]
                h_d = h["donor"][d]
                z = model.head_donor.stance(h_d, topic_hard[idx], term_ids[idx])

                # Weight by amount
                w = torch.ones_like(z)
                if ea is not None and ea.numel() > 0 and ea.size(0) == ei.size(1):
                    amt = ea[:, cfg.exp_amount_col].float().clamp(min=0.0)[m]
                    w = torch.log1p(amt).clamp(min=0.0, max=float(cfg.out_exp_amt_clip))
                logits.index_add_(0, idx, float(cfg.out_exp_w) * (z * w))

    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss * float(cfg.outcome_weight)


def regularize_factors(model, cfg):
    """L2 regularization on factor parameters."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for head in [model.head_leg, model.head_donor, model.head_lobby]:
        for p in head.parameters():
            reg = reg + float(cfg.l2_weight) * (p.pow(2).mean())
    return reg


@torch.no_grad()
def export_actor_topic_tables(model, data, cfg, out_path, device):
    model.eval()
    num_topics = int(model.num_topics)
    rows = []

    def _export_actor(actor_type, head):
        if actor_type not in data.node_types:
            return
        n = int(data[actor_type].num_nodes)
        if n == 0:
            return

        loader = NeighborLoader(
            data,
            input_nodes=(actor_type, torch.arange(n)),
            num_neighbors=list(cfg.num_neighbors_actor_export),
            batch_size=int(cfg.actor_batch_size),
            shuffle=False,
            persistent_workers=False,
        )

        topic_ids = torch.arange(num_topics, device=device)
        term_ids = torch.zeros_like(topic_ids)

        for batch in tqdm(loader, desc=f"export {actor_type}"):
            batch = batch.to(device)
            h = model.encoder(batch)
            bs = int(batch[actor_type].batch_size)
            h_a = h[actor_type][:bs]
            global_ids = batch[actor_type].n_id[:bs].cpu().numpy()

            h_exp = h_a.repeat_interleave(num_topics, dim=0)
            t_exp = topic_ids.repeat(bs)
            term_exp = term_ids.repeat(bs)

            z = head.stance(h_exp, t_exp, term_exp).view(bs, num_topics)
            infl = head.influence(h_exp, t_exp, term_exp).view(bs, num_topics)

            infl_norm = infl / (infl.sum(dim=1, keepdim=True) + 1e-8)
            impact = z * infl_norm

            df = pd.DataFrame(
                {
                    "actor_type": actor_type,
                    "actor_index": np.repeat(global_ids, num_topics),
                    "topic_id": np.tile(np.arange(num_topics), bs),
                    "stance": z.cpu().numpy().reshape(-1),
                    "influence": infl.cpu().numpy().reshape(-1),
                    "influence_norm": infl_norm.cpu().numpy().reshape(-1),
                    "impact": impact.cpu().numpy().reshape(-1),
                }
            )
            rows.append(df)

    _export_actor("legislator_term", model.head_leg)
    _export_actor("donor", model.head_donor)
    _export_actor("lobby_firm", model.head_lobby)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(out_path, index=False)


def infer_edge_dims(data, et):
    if et not in data.edge_types:
        return 0
    ea = getattr(data[et], "edge_attr", None)
    return int(ea.size(-1)) if ea is not None and ea.numel() > 0 else 0


def train(cfg):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    ckpt_dir = ensure_dir(os.path.join(cfg.out_dir, "checkpoints"))
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")

    data = torch.load(cfg.data_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise RuntimeError("data_path did not load HeteroData")

    data = preprocess_data(data, standardize_x=cfg.standardize_x, keep_bidir=True)

    bill_topic = get_bill_topic_tensor(data)
    num_topics = int(bill_topic[bill_topic >= 0].max().item() + 1)

    bill_term_raw = get_bill_term_tensor(data)
    bill_term_idx_full, term_to_idx, term_values = build_term_index(bill_term_raw)
    num_terms = int(len(term_to_idx))

    vote_edge_dim = infer_edge_dims(
        data, ("legislator_term", "voted_on", "bill_version")
    )
    exp_edge_dim = infer_edge_dims(data, ("donor", "expenditure", "bill"))
    donate_edge_dim = infer_edge_dims(data, ("donor", "donated_to", "legislator_term"))
    lobby_edge_dim = infer_edge_dims(data, ("lobby_firm", "lobbied", "legislator_term"))

    model = LegislativeStanceModel(
        data=data,
        num_topics=num_topics,
        num_terms=num_terms,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        rank=cfg.rank,
        use_soft_topics=cfg.use_soft_topics,
        vote_edge_dim=vote_edge_dim,
        exp_edge_dim=exp_edge_dim,
        donate_edge_dim=donate_edge_dim,
        lobby_edge_dim=lobby_edge_dim,
    )

    dev = (
        device_default(cfg.device_preference)
        if cfg.device is None
        else torch.device(cfg.device)
    )
    model = model.to(dev)

    if cfg.export_only:
        ckpt = torch.load(
            os.path.join(ckpt_dir, cfg.export_ckpt_name), map_location="cpu"
        )
        model.load_state_dict(ckpt["model"], strict=True)
        out_path = os.path.join(cfg.out_dir, cfg.export_name)
        export_actor_topic_tables(model, data, cfg, out_path, dev)
        print(f"Saved actor-topic parquet: {out_path}")
        return

    if cfg.start_from_ckpt:
        ckpt = torch.load(
            os.path.join(ckpt_dir, cfg.export_ckpt_name), map_location="cpu"
        )
        model.load_state_dict(ckpt["model"], strict=True)

    vote_loader = make_vote_loader(data, cfg)
    exp_loader = (
        make_expenditure_loader(data, cfg)
        if ("donor", "expenditure", "bill") in data.edge_types
        else None
    )
    bill_loader = make_bill_loader(data, cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    vote_iter = iter(vote_loader)
    exp_iter = iter(exp_loader) if exp_loader is not None else None
    bill_iter = iter(bill_loader)

    def next_batch(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    with open(metrics_path, "a", encoding="utf-8") as mf:
        for epoch in range(int(cfg.epochs)):
            model.train()
            losses = {
                "vote": [],
                "exp_s": [],
                "exp_m": [],
                "topic": [],
                "smooth": [],
                "out": [],
                "reg": [],
                "total": [],
            }

            pbar = tqdm(range(int(cfg.steps_per_epoch)), desc=f"train e{epoch}")

            for step in pbar:
                opt.zero_grad(set_to_none=True)
                total = torch.tensor(0.0, device=dev)

                do_vote = True
                do_exp = (
                    exp_loader is not None
                    and (cfg.exp_stance_weight > 0 or cfg.exp_amount_weight > 0)
                    and step % int(cfg.exp_every) == 0
                )
                do_bill = (
                    cfg.topic_ce_weight > 0 or cfg.outcome_weight > 0
                ) and step % int(cfg.bill_every) == 0

                # Select batch
                if do_vote:
                    batch, vote_iter = next_batch(vote_iter, vote_loader)
                    batch = batch.to(dev)
                elif do_exp:
                    batch, exp_iter = next_batch(exp_iter, exp_loader)
                    batch = batch.to(dev)
                else:
                    batch, bill_iter = next_batch(bill_iter, bill_loader)
                    batch = batch.to(dev)

                # Select active edge types
                active = set()
                if do_vote:
                    active.add(("legislator_term", "voted_on", "bill_version"))
                if do_exp:
                    active.add(("donor", "expenditure", "bill"))
                if do_bill:
                    active.add(("bill_version", "priorVersion", "bill_version"))
                if not active:
                    active = set(batch.edge_types)

                h = model.encoder(batch, active_edge_types=active)

                # Compute losses
                if do_vote:
                    lv = vote_direction_loss(model, batch, h, bill_term_idx_full, cfg)
                    ls = version_smooth_loss(model, batch, h, cfg)
                    lr = regularize_factors(model, cfg)
                    total = total + lv + ls + lr
                    losses["vote"].append(float(lv.item()))
                    losses["smooth"].append(float(ls.item()))
                    losses["reg"].append(float(lr.item()))

                if do_exp:
                    les, lem = exp_stance_and_amount_loss(
                        model, batch, h, bill_term_idx_full, cfg
                    )
                    ls = version_smooth_loss(model, batch, h, cfg)
                    lr = regularize_factors(model, cfg)
                    total = total + les + lem + ls + lr
                    losses["exp_s"].append(float(les.item()))
                    losses["exp_m"].append(float(lem.item()))
                    losses["smooth"].append(float(ls.item()))
                    losses["reg"].append(float(lr.item()))

                if do_bill:
                    lt = bill_topic_mixture_loss(model, batch, h, cfg)
                    lo = outcome_aux_loss(model, batch, h, bill_term_idx_full, cfg)
                    ls = version_smooth_loss(model, batch, h, cfg)
                    lr = regularize_factors(model, cfg)
                    total = total + lt + lo + ls + lr
                    losses["topic"].append(float(lt.item()))
                    losses["out"].append(float(lo.item()))
                    losses["smooth"].append(float(ls.item()))
                    losses["reg"].append(float(lr.item()))

                if not torch.isfinite(total):
                    continue

                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()

                losses["total"].append(float(total.item()))
                pbar.set_postfix(
                    {
                        "tot": (
                            float(np.mean(losses["total"][-50:]))
                            if losses["total"]
                            else 0.0
                        ),
                        "vote": (
                            float(np.mean(losses["vote"][-50:]))
                            if losses["vote"]
                            else 0.0
                        ),
                        "expS": (
                            float(np.mean(losses["exp_s"][-50:]))
                            if losses["exp_s"]
                            else 0.0
                        ),
                        "bill": (
                            float(np.mean(losses["out"][-50:]))
                            if losses["out"]
                            else 0.0
                        ),
                    }
                )

            rec = {
                "ts": time.time(),
                "epoch": int(epoch),
                "loss_total": (
                    float(np.mean(losses["total"])) if losses["total"] else None
                ),
                "loss_vote": float(np.mean(losses["vote"])) if losses["vote"] else None,
                "loss_exp_stance": (
                    float(np.mean(losses["exp_s"])) if losses["exp_s"] else None
                ),
                "loss_exp_amount": (
                    float(np.mean(losses["exp_m"])) if losses["exp_m"] else None
                ),
                "loss_topic": (
                    float(np.mean(losses["topic"])) if losses["topic"] else None
                ),
                "loss_outcome": (
                    float(np.mean(losses["out"])) if losses["out"] else None
                ),
                "loss_smooth": (
                    float(np.mean(losses["smooth"])) if losses["smooth"] else None
                ),
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
                    "num_topics": int(num_topics),
                    "num_terms": int(num_terms),
                    "term_values": term_values.tolist(),
                },
                ckpt_path,
            )

            print(
                f"[epoch {epoch}] total={rec['loss_total']:.5f} "
                f"vote={rec['loss_vote']} expS={rec['loss_exp_stance']} out={rec['loss_outcome']}"
            )

    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                k: getattr(cfg, k) for k in cfg.__dict__.keys() if not k.startswith("_")
            },
            "num_topics": int(num_topics),
            "num_terms": int(num_terms),
            "term_values": term_values.tolist(),
        },
        final_path,
    )
    print(f"Saved final checkpoint: {final_path}")

    if cfg.export_after_train:
        out_path = os.path.join(cfg.out_dir, cfg.export_name)
        export_actor_topic_tables(model, data, cfg, out_path, dev)
        print(f"Saved actor-topic parquet: {out_path}")


@dataclass
class Config:
    data_path: str = "data5.pt"
    out_dir: str = "dashboard/backend/data/outs"
    seed: int = 13

    device_preference: str = "mps"
    device: str = None

    d_model: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    rank: int = 24

    standardize_x: bool = True
    use_soft_topics: bool = True

    epochs: int = 15
    lr: float = 2e-4
    wd: float = 1e-3
    grad_clip: float = 1.0
    steps_per_epoch: int = 24

    vote_edge_batch_size: int = 6000
    exp_edge_batch_size: int = 512
    bill_batch_size: int = 1024
    actor_batch_size: int = 512

    num_neighbors_vote: list = field(default_factory=lambda: [14, 7])
    num_neighbors_exp: list = field(default_factory=lambda: [20, 12])
    num_neighbors_bill: list = field(default_factory=lambda: [12, 8])
    num_neighbors_actor_export: list = field(default_factory=lambda: [25, 12])

    vote_target_col: int = -1
    exp_stance_col: int = -1
    exp_amount_col: int = 0

    vote_dir_weight: float = 1.0
    vote_label_smooth: float = 0.03

    exp_stance_weight: float = 0.25
    exp_amount_weight: float = 0.05

    topic_ce_weight: float = 0.15
    topic_label_smooth: float = 0.05

    version_smooth_weight: float = 0.05

    outcome_weight: float = 0.5
    out_lam_w: float = 0.35
    out_kap_w: float = 0.15
    out_vote_w: float = 0.1
    out_exp_w: float = 0.01
    out_exp_amt_clip: float = 12.0

    amt_a0: float = 0.0
    amt_a1: float = 0.20
    amt_a2: float = 0.35
    amt_a3: float = 0.12
    amt_a4: float = 0.05

    l2_weight: float = 1e-5

    exp_every: int = 3
    bill_every: int = 4

    export_after_train: bool = True
    export_only: bool = False
    start_from_ckpt: bool = False
    export_ckpt_name: str = "final.pt"
    export_name: str = "actor_topic.parquet"

    export_topic_block: int = 16


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)
    train(cfg)


if __name__ == "__main__":
    main()
