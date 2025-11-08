import os, json, gc, random, warnings, torch, numpy as np, pandas as pd
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)


def _dev():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CFG:
    seed = 42
    d = 164
    layers = 2
    drop = 0.1
    lr = 1e-3
    wd = 1e-4
    epochs = 7
    bsz = 1024
    neigh_budgets = {
        "bill_version": [32, 12, 6],
        "bill": [32, 12, 6],
        "legislator_term": [16, 8, 4],
        "committee": [16, 8, 4],
        "lobby_firm": [12, 8, 4],
        "donor": [12, 8, 4],
    }
    lambda_outcome = 1.0
    lambda_contrast = 0.15
    lambda_temporal = 0.08
    lambda_actor_topic = 1.0
    save_dir = "dashboard/backend/data/outs"
    max_snapshots = 10
    contrastive_tau = 0.07
    num_workers = 0
    actor_types = ("legislator_term", "committee", "donor", "lobby_firm")
    infer_types = (
        "bill",
        "legislator",
        "legislator_term",
        "committee",
        "donor",
        "lobby_firm",
        "bill_version",
    )
    msg_chunk_edges = 30000


ACCUM_STEPS = 2
DEVICE = _dev()
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

MANUAL_EDGE_TCOLS = {
    ("legislator_term", "voted_on", "bill_version"): [0],
    ("committee", "rev_read", "bill_version"): [-1],
    ("donor", "donated_to", "legislator_term"): [0],
    ("lobby_firm", "lobbied", "legislator_term"): [0],
    ("lobby_firm", "lobbied", "committee"): [0],
    ("bill_version", "read", "committee"): [-1],
}


def empty_cache_mps():
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass


def load_hetero(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_bidirectional(data: HeteroData):
    for s, r, t in list(data.edge_types):
        e = data[(s, r, t)]
        rev = (t, r + "_rev", s)
        if rev not in data.edge_types:
            if e.edge_index.numel() == 0:
                data[rev].edge_index = e.edge_index.flip(0)
            else:
                src, dst = e.edge_index
                data[rev].edge_index = torch.stack([dst, src], dim=0)
                if (
                    "edge_attr" in e
                    and e.edge_attr is not None
                    and e.edge_attr.size(0) == src.size(0)
                ):
                    data[rev].edge_attr = e.edge_attr.clone()


def normalize_node_features(data):
    for nt in data.node_types:
        if "x" in data[nt]:
            x = data[nt].x
            if x is None or x.numel() == 0:
                continue
            if x.dtype not in (torch.float16, torch.float32, torch.float64):
                continue
            x = x.clone()
            mask = torch.isfinite(x)
            if not mask.all():
                x[~mask] = 0.0
            m = x.mean(0, keepdim=True)
            v = x.var(0, unbiased=False, keepdim=True).clamp_min(1e-8)
            data[nt].x = (x - m) / torch.sqrt(v)


def get_edge_time_attr(data, etype, default_t=0.0):
    try:
        e = data[etype]
        E = e.edge_index.size(1)
        if "edge_attr" in e and e.edge_attr is not None and e.edge_attr.size(0) == E:
            cols = MANUAL_EDGE_TCOLS.get(etype, None)
            if cols is not None and len(cols) > 0:
                return e.edge_attr[:, cols[0]].float()
        for key in ("time", "timestamp", "date"):
            if key in e:
                v = e[key]
                if isinstance(v, torch.Tensor) and v.numel() == E:
                    return v.float()
    except:
        pass
    return torch.full((data[etype].edge_index.size(1),), float(default_t))


def build_time_slices(data):
    caps_per_edge = 150000
    cap_total = 600000
    ts = []
    for et in data.edge_types:
        t = get_edge_time_attr(data, et)
        if t.numel() == 0:
            continue
        t = t.detach().cpu().float()
        n = t.numel()
        if n > caps_per_edge:
            idx = torch.randint(0, n, (caps_per_edge,))
            t = t[idx]
        ts.append(t)
    if len(ts) == 0:
        return [None]
    all_t = torch.cat(ts)
    if all_t.numel() > cap_total:
        idx = torch.randint(0, all_t.numel(), (cap_total,))
        all_t = all_t[idx]
    arr = all_t.numpy()
    qs = np.quantile(arr, np.linspace(0.0, 1.0, CFG.max_snapshots + 1))
    return [(float(qs[s]), float(qs[s + 1])) for s in range(CFG.max_snapshots)]


def filter_graph_by_time(data, time_window):
    if time_window is None:
        return data
    lo, hi = time_window
    out = HeteroData()
    for nt in data.node_types:
        out[nt].num_nodes = data[nt].num_nodes
        for f in data[nt].keys():
            out[nt][f] = data[nt][f]
    for et in data.edge_types:
        eidx = data[et].edge_index
        if eidx.numel() == 0:
            out[et].edge_index = eidx
            for f in data[et].keys():
                if f != "edge_index":
                    out[et][f] = data[et][f]
            continue
        t = get_edge_time_attr(data, et)
        if t.numel() == 0:
            out[et].edge_index = eidx
            for f in data[et].keys():
                if f != "edge_index":
                    out[et][f] = data[et][f]
            continue
        mask = (t >= lo) & (t <= hi)
        keep = torch.where(mask)[0]
        out[et].edge_index = eidx[:, keep]
        for f in data[et].keys():
            if f == "edge_index":
                continue
            val = data[et][f]
            if isinstance(val, torch.Tensor) and val.size(0) == eidx.size(1):
                out[et][f] = val[keep]
            else:
                out[et][f] = val
    return out


def per_type_laplacian_pe(data):
    pe = {}
    for nt in data.node_types:
        pe[nt] = torch.zeros(data[nt].num_nodes, 0)
    return pe


class TypeLinear(nn.Module):
    def __init__(self, in_dims, d, drop, pe_dims, base_embeds):
        super().__init__()
        self.ts = list(in_dims.keys())
        self.pe_dims = pe_dims
        self.base = nn.ModuleDict(base_embeds)
        self.lins = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Linear(in_dims[t] + pe_dims.get(t, 0), d),
                    nn.ReLU(),
                    nn.Dropout(drop),
                )
                for t in self.ts
            }
        )

    def forward(self, batch, pedict):
        out = {}
        for t in self.ts:
            if "x" in batch[t]:
                x = batch[t].x
            else:
                n = batch[t].num_nodes
                x = self.base[t](torch.arange(n, device=DEVICE))
            pe = pedict.get(t, None)
            if pe is not None:
                if pe.size(0) != x.size(0):
                    pe = torch.zeros(
                        x.size(0), pe.size(1), device=x.device, dtype=x.dtype
                    )
                x = torch.cat([x, pe.to(x.device, dtype=x.dtype)], dim=-1)
            lin0 = self.lins[t][0]
            exp = lin0.in_features
            cur = x.size(1)
            if cur < exp:
                pad = torch.zeros(x.size(0), exp - cur, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
            elif cur > exp:
                x = x[:, :exp]
            if x.size(0) == 0:
                out[t] = torch.zeros(
                    0, lin0.out_features, device=x.device, dtype=x.dtype
                )
            else:
                out[t] = self.lins[t](x)
        return out


class RelAttnConv(nn.Module):
    def __init__(self, d, n_heads=2, drop=0.1, edge_dim=0, time_dim=8):
        super().__init__()
        self.d = d
        self.h = n_heads
        self.dk = d // n_heads
        self.drop = nn.Dropout(drop)
        self.time_pe = nn.Linear(1, time_dim) if time_dim > 0 else None
        self.rel_film = nn.ModuleDict()
        self.rel_bias = nn.ParameterDict()
        self.edge_dim = edge_dim
        self.time_dim = time_dim

    def _rel_params(self, key, device):
        if key not in self.rel_film:
            in_dim = self.edge_dim + (self.time_dim if self.time_dim > 0 else 0)
            self.rel_film[key] = nn.Sequential(
                nn.Linear(in_dim, self.d // 2),
                nn.ReLU(),
                nn.Linear(self.d // 2, 2 * self.d),
            ).to(device)
            self.rel_bias[key] = nn.Parameter(torch.zeros(self.h, device=device))

    def forward(self, q, k, v, edge_index, edge_attr=None, edge_time=None, relkey=None):
        if edge_index.numel() == 0 or q.size(0) == 0:
            return torch.zeros_like(q)
        device = q.device
        dtype = q.dtype
        self._rel_params(relkey, device)
        src, dst = edge_index
        Q = q.view(-1, self.h, self.dk).to(dtype)
        K = k.view(-1, self.h, self.dk).to(dtype)
        V = v.view(-1, self.h, self.dk).to(dtype)
        out_acc = torch.zeros(q.size(0), self.h * self.dk, device=device, dtype=dtype)
        step = CFG.msg_chunk_edges
        rf = self.rel_film[relkey]
        rb = self.rel_bias[relkey]

        for start in range(0, src.numel(), step):
            end = min(start + step, src.numel())
            s = src[start:end]
            d = dst[start:end]
            qh = Q[d]
            kh = K[s]
            vh = V[s]

            if edge_attr is not None or edge_time is not None:
                feats = []
                if edge_attr is not None:
                    feats.append(edge_attr[start:end].to(device=device, dtype=dtype))
                if edge_time is not None:
                    t = edge_time[start:end].to(device).view(-1, 1).float()
                    t = self.time_pe(t) if self.time_pe is not None else t
                    feats.append(t.to(dtype))
                if feats:
                    feat = torch.cat(feats, dim=-1)
                    gb = rf(feat).to(dtype)
                    gamma, beta = gb.chunk(2, dim=-1)
                    gamma = gamma.view(-1, self.h, self.dk).tanh()
                    beta = beta.view(-1, self.h, self.dk).tanh()
                    vh = vh * (1 + gamma) + beta

            score = (qh * kh).sum(-1) / (self.dk**0.5) + rb.unsqueeze(0)
            score = score.to(dtype)
            att = torch.softmax(score, dim=0)
            m = (att.unsqueeze(-1) * vh).reshape(-1, self.h * self.dk).to(dtype)
            out_acc.index_add_(0, d, m)

        return self.drop(out_acc)


class RelationalAttnBackbone(nn.Module):
    def __init__(self, metadata, d, layers, drop, edge_dims):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.q_proj = nn.ModuleDict(
            {nt: nn.Linear(d, d, bias=False) for nt in self.node_types}
        )
        self.k_proj = nn.ModuleDict(
            {nt: nn.Linear(d, d, bias=False) for nt in self.node_types}
        )
        self.v_proj = nn.ModuleDict(
            {nt: nn.Linear(d, d, bias=False) for nt in self.node_types}
        )
        self.o_proj = nn.ModuleDict(
            {nt: nn.Linear(d, d, bias=True) for nt in self.node_types}
        )
        self.layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.ModuleDict()
            for s, r, t in self.edge_types:
                key = f"{s}|{r}|{t}"
                e_dim = edge_dims.get((s, r, t), 0)
                layer[key] = RelAttnConv(
                    d, n_heads=2, drop=drop, edge_dim=e_dim, time_dim=8
                )
            self.layers.append(layer)
        self.layer_drop = nn.Parameter(torch.tensor(0.15))

    def forward(self, h, batch):
        x = h
        for layer in self.layers:
            new = {nt: torch.zeros_like(x[nt]) for nt in x}
            q = {nt: self.q_proj[nt](x[nt]) for nt in x}
            k = {nt: self.k_proj[nt](x[nt]) for nt in x}
            v = {nt: self.v_proj[nt](x[nt]) for nt in x}
            for s, r, t in batch.edge_types:
                key = f"{s}|{r}|{t}"
                if key not in layer:
                    continue
                conv = layer[key]
                e = batch[(s, r, t)]
                eattr = (
                    e.edge_attr
                    if ("edge_attr" in e and e.edge_attr is not None)
                    else None
                )
                etime = get_edge_time_attr(batch, (s, r, t)).to(x[s].device)
                msg = conv(
                    q[t],
                    k[s],
                    v[s],
                    e.edge_index,
                    edge_attr=eattr,
                    edge_time=etime,
                    relkey=key,
                )
                new[t] += F.relu(self.o_proj[t](msg))

            if self.training and torch.rand(()) < self.layer_drop.sigmoid():
                x = x
            else:
                x = new
        return x


class OutcomeHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 3)
        )

    def forward(self, bill_emb):
        return self.mlp(bill_emb)


class VoteHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 3)
        )

    def forward(self, src, dst):
        return self.mlp(torch.cat([src, dst], dim=-1))


class GatekeepHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, bv, cmte):
        return self.mlp(torch.cat([bv, cmte], dim=-1))


class ContrastiveProj(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.p = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return F.normalize(self.p(x), dim=-1)


class ActorTopicHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.stance = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.conf = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1), nn.Softplus()
        )

    def forward(self, actor_emb, topic_proto):
        z = actor_emb * topic_proto
        s = torch.tanh(self.stance(z)).squeeze(-1)
        c = self.conf(z).squeeze(-1)
        return s, c


class InfluenceHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1), nn.Softplus()
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class TopicPrototypes(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        self.protos = nn.Parameter(F.normalize(torch.randn(K, d), dim=-1))

    @torch.no_grad()
    def set_from_embeddings(self, bill_cluster, bill_emb, K):
        if bill_cluster is None or bill_emb is None:
            return
        e = F.normalize(bill_emb, dim=-1)
        for t in range(K):
            mask = bill_cluster == t
            if mask.any():
                self.protos.data[t] = F.normalize(e[mask].mean(0), dim=-1)
            else:
                self.protos.data[t] = F.normalize(self.protos.data[t], dim=-1)

    def get(self):
        return F.normalize(self.protos, dim=-1)


class Model(nn.Module):
    def __init__(self, data, d, layers, drop, pe, K):
        super().__init__()
        in_dims = {}
        base_embeds = {}
        for nt in data.node_types:
            if "x" in data[nt]:
                in_dims[nt] = int(data[nt].x.size(1))
            else:
                in_dims[nt] = d
                base_embeds[nt] = nn.Embedding(int(data[nt].num_nodes), d)
        pe_dims = {
            nt: (pe[nt].size(1) if (pe is not None and nt in pe) else 0)
            for nt in data.node_types
        }
        self.enc = TypeLinear(in_dims, d, drop, pe_dims, base_embeds)
        edge_dims = {}
        for s, r, t in data.edge_types:
            e = data[(s, r, t)]
            edge_dims[(s, r, t)] = (
                e.edge_attr.size(1)
                if "edge_attr" in e and e.edge_attr is not None
                else 0
            )
        self.backbone = RelationalAttnBackbone(
            data.metadata(), d, layers, drop, edge_dims
        )
        self.outcome = OutcomeHead(d)
        self.vote = VoteHead(d)
        self.gatekeep = GatekeepHead(d)
        self.cproj = ContrastiveProj(d)
        self.actor_topic = ActorTopicHead(d)
        self.influence = InfluenceHead(d)
        self.d = d
        self.pe = pe
        self.pe_dims = pe_dims
        self.topic_bank = TopicPrototypes(K, d)
        self.actor_types = CFG.actor_types

    def forward(self, batch):
        pedict = {}
        for nt in batch.node_types:
            exp = self.pe_dims.get(nt, 0)
            if exp > 0:
                if (
                    (self.pe is not None)
                    and (nt in self.pe)
                    and (self.pe[nt] is not None)
                    and (self.pe[nt].numel() > 0)
                ):
                    if hasattr(batch[nt], "n_id"):
                        idx = batch[nt].n_id.to(torch.long, non_blocking=True)
                        pe_nt = self.pe[nt][idx.cpu()].to(DEVICE)
                    else:
                        pe_nt = self.pe[nt].to(DEVICE)
                    if pe_nt.size(0) != batch[nt].num_nodes:
                        pe_nt = torch.zeros(batch[nt].num_nodes, exp, device=DEVICE)
                else:
                    pe_nt = torch.zeros(batch[nt].num_nodes, exp, device=DEVICE)
                pedict[nt] = pe_nt
        h0 = self.enc(batch, pedict)
        h = self.backbone(h0, batch)
        return h


def _node_attr(batch, nt, key, expected_len):
    if key not in batch[nt]:
        return None
    v = batch[nt][key]
    if (
        isinstance(v, torch.Tensor)
        and v.size(0) != expected_len
        and hasattr(batch[nt], "n_id")
    ):
        v = v[batch[nt].n_id]
    return v


def outcome_loss(model, h, batch):
    if "bill" not in batch.node_types:
        return torch.tensor(0.0, device=DEVICE)
    y = _node_attr(batch, "bill", "y", h["bill"].size(0))
    if y is None:
        return torch.tensor(0.0, device=DEVICE)
    y = y.to(DEVICE).long()
    logits = model.outcome(h["bill"])
    return F.cross_entropy(logits, y, ignore_index=-1)


def _paired_bill_bv_indices(batch):
    if ("bill_version", "is_version", "bill") not in batch.edge_types:
        return None
    e = batch[("bill_version", "is_version", "bill")].edge_index
    if e.numel() == 0:
        return None
    bv = e[0]
    b = e[1]
    n = min(bv.size(0), b.size(0))
    return bv[:n], b[:n]


def contrastive_loss(model, h, batch, tau):
    if "bill" not in h or "bill_version" not in h:
        return torch.tensor(0.0, device=DEVICE)
    pairs = _paired_bill_bv_indices(batch)
    if pairs is None:
        return torch.tensor(0.0, device=DEVICE)
    bv_idx, b_idx = pairs
    if bv_idx.numel() < 2:
        return torch.tensor(0.0, device=DEVICE)
    a = model.cproj(h["bill_version"][bv_idx.to(DEVICE)])
    b = model.cproj(h["bill"][b_idx.to(DEVICE)])
    n = min(a.size(0), b.size(0), 4096)
    a = a[:n]
    b = b[:n]
    sim = a @ b.t()
    targets = torch.arange(n, device=DEVICE)
    return 0.5 * (
        F.cross_entropy(sim / tau, targets) + F.cross_entropy(sim.t() / tau, targets)
    )


def temporal_smooth_loss(h_prev, h_curr, keys):
    if h_prev is None:
        return torch.tensor(0.0, device=DEVICE)
    ks = []
    for k in keys:
        if (
            k in h_prev
            and k in h_curr
            and h_prev[k].size(0) > 0
            and h_curr[k].size(0) > 0
        ):
            n = min(h_prev[k].size(0), h_curr[k].size(0))
            ks.append(F.mse_loss(h_curr[k][:n], h_prev[k][:n]))
    if len(ks) == 0:
        return torch.tensor(0.0, device=DEVICE)
    return torch.stack(ks).mean()


def vote_loss(model, h, batch):
    et = ("legislator_term", "voted_on", "bill_version")
    if et not in batch.edge_types:
        return torch.tensor(0.0, device=DEVICE)
    e = batch[et]
    if e.edge_index.numel() == 0 or e.edge_attr is None:
        return torch.tensor(0.0, device=DEVICE)
    y = e.edge_attr[:, 0].to(DEVICE).long() + 1
    logits = model.vote(
        h["legislator_term"][e.edge_index[0]], h["bill_version"][e.edge_index[1]]
    )
    cw = torch.tensor([1.0, 0.6, 1.0], device=DEVICE)
    return F.cross_entropy(logits, y, weight=cw)


def gatekeep_labels(batch):
    et = ("bill_version", "read", "committee")
    if et not in batch.edge_types:
        return None
    e = batch[et]
    if e.edge_index.numel() == 0:
        return None
    if "advance" in e:
        return e
    return None


def gatekeep_loss(model, h, batch):
    e = gatekeep_labels(batch)
    if e is None:
        return torch.tensor(0.0, device=DEVICE)
    logits = model.gatekeep(
        h["bill_version"][e.edge_index[0]], h["committee"][e.edge_index[1]]
    )
    y = e.advance.to(DEVICE).long()
    return F.cross_entropy(logits, y)


def actor_topic_signals(batch, h, topic_protos):
    if ("legislator_term", "voted_on", "bill_version") not in batch.edge_types:
        return None
    vote = batch[("legislator_term", "voted_on", "bill_version")]
    pair = (
        batch[("bill_version", "is_version", "bill")]
        if ("bill_version", "is_version", "bill") in batch.edge_types
        else None
    )
    if vote.edge_index.numel() == 0 or pair is None or pair.edge_index.numel() == 0:
        return None
    lt = vote.edge_index[0].to(DEVICE)
    bv = vote.edge_index[1].to(DEVICE)
    bv2b = torch.full(
        (batch["bill_version"].num_nodes,), -1, device=DEVICE, dtype=torch.long
    )
    bv2b[pair.edge_index[0].to(DEVICE)] = pair.edge_index[1].to(DEVICE)
    b = bv2b[bv]
    keep = b >= 0
    if keep.sum() == 0:
        return None
    lt, b = lt[keep], b[keep]
    vote_val = vote.edge_attr[:, 0].to(DEVICE).float()[keep]
    cluster = _node_attr(batch, "bill", "cluster", h["bill"].size(0))
    if cluster is None:
        return None
    topic = cluster.to(DEVICE).long()[b]
    a_emb = h["legislator_term"][lt]
    t_proto = topic_protos[topic]
    y = vote_val.clamp(-1, 1)
    w = vote_val.abs().clamp_min(0.3)
    return a_emb, t_proto, y, w, lt, topic


def actor_topic_loss_new(model, batch, h, protos):
    sig = actor_topic_signals(batch, h, protos)
    if sig is None:
        return torch.tensor(0.0, device=DEVICE)
    a_emb, t_proto, y, w, lt, topic = sig
    s, c = model.actor_topic(a_emb, t_proto)
    reg = F.mse_loss(s, y, reduction="none")
    reg = (reg * w).mean()
    perm = torch.randperm(lt.size(0), device=DEVICE)
    i, j = lt, lt[perm]
    mask = (i == j) & (y != y[perm])
    if mask.any():
        diff = s[mask] - s[perm][mask]
        sign = (y[mask] - y[perm][mask]).sign()
        rank_loss = F.relu(1.0 - diff * sign).mean()
    else:
        rank_loss = torch.tensor(0.0, device=DEVICE)
    return 0.7 * reg + 0.3 * rank_loss


def prepare_num_neighbors(data):
    depth = CFG.layers
    out = {}
    for et in data.edge_types:
        dst = et[2]
        budget = CFG.neigh_budgets.get(dst, [16, 8, 4])
        if len(budget) < depth:
            budget = budget + [budget[-1]] * (depth - len(budget))
        elif len(budget) > depth:
            budget = budget[:depth]
        out[et] = budget
    return out


def subsample_edges_epoch(data: HeteroData, caps):
    out = HeteroData()
    for nt in data.node_types:
        out[nt].num_nodes = data[nt].num_nodes
        for f in data[nt].keys():
            out[nt][f] = data[nt][f]
    g = torch.Generator().manual_seed(torch.randint(0, 10_000, (1,)).item())
    for et in data.edge_types:
        e = data[et]
        E = e.edge_index.size(1)
        cap = caps.get(et, None)
        if cap is None or E <= cap:
            out[et] = e
            continue
        idx = torch.randperm(E, generator=g)[:cap]
        out[et].edge_index = e.edge_index[:, idx]
        for k, v in e.items():
            if k == "edge_index":
                continue
            if isinstance(v, torch.Tensor) and v.size(0) == E:
                out[et][k] = v[idx]
            else:
                out[et][k] = v
    return out


@torch.no_grad()
def infer_embeddings(model, data, bsz):
    model.eval()
    out_dtype = next(model.parameters()).dtype
    embs = {}
    for nt in CFG.infer_types:
        if nt not in data.node_types:
            continue
        embs[nt] = torch.zeros(
            data[nt].num_nodes, model.d, device=DEVICE, dtype=out_dtype
        )
    for nt in CFG.infer_types:
        if nt not in data.node_types:
            continue
        input_nodes = (nt, torch.arange(data[nt].num_nodes))
        num_neighbors = {et: [-1] * CFG.layers for et in data.edge_types}
        loader = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=num_neighbors,
            batch_size=bsz,
            shuffle=False,
            num_workers=CFG.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )
        for batch in tqdm(loader, desc=f"infer {nt}", leave=False):
            batch = batch.to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.bfloat16, enabled=True
            ):
                h = model(batch)
            gidx = batch[nt].n_id.to(DEVICE)
            embs[nt][gidx] = h[nt].to(out_dtype)
            del h, batch
            gc.collect()
            empty_cache_mps()
    for nt in embs:
        embs[nt] = F.normalize(embs[nt].float(), dim=-1).cpu().numpy()
    return embs


def compute_topic_protos_from_full(data, emb):
    if "bill" not in emb:
        return torch.randn(1, CFG.d, device=DEVICE)
    if "cluster" not in data["bill"]:
        return torch.randn(1, CFG.d, device=DEVICE)
    k = torch.as_tensor(data["bill"].cluster, dtype=torch.long, device=DEVICE)
    e = torch.from_numpy(emb["bill"]).to(DEVICE).float()
    K = int(torch.max(k).item()) + 1
    protos = torch.zeros(K, e.size(1), device=DEVICE)
    for t in range(K):
        mask = k == t
        if mask.any():
            protos[t] = F.normalize(e[mask].mean(0), dim=-1)
        else:
            protos[t] = F.normalize(torch.randn_like(protos[t]), dim=-1)
    return protos


def vote_loss_weight():
    return 0.8


def gatekeep_loss_weight():
    return 0.6


def train_one_snapshot(model, data, optimizer, sid, protos):
    model.train()
    caps = {
        ("legislator_term", "voted_on", "bill_version"): 1200000,
        ("bill_version", "read", "committee"): 40000,
        ("donor", "donated_to", "legislator_term"): 2000,
        ("lobby_firm", "lobbied", "committee"): 3000,
        ("lobby_firm", "lobbied", "legislator_term"): 180,
    }
    data_e = subsample_edges_epoch(data, caps)
    base_nt = "bill" if "bill" in data_e.node_types else list(data_e.node_types)[0]
    input_nodes = (base_nt, torch.arange(data_e[base_nt].num_nodes))
    num_neighbors = prepare_num_neighbors(data_e)
    loader = NeighborLoader(
        data_e,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=CFG.bsz,
        shuffle=True,
        num_workers=CFG.num_workers,
        persistent_workers=False,
        pin_memory=False,
    )
    prev = None
    keys = [
        "legislator_term",
        "committee",
        "donor",
        "lobby_firm",
        "bill",
        "bill_version",
    ]
    total = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(
        tqdm(loader, desc=f"snapshot {sid} train", leave=False)
    ):
        batch = batch.to(DEVICE, non_blocking=False)
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.bfloat16, enabled=True
        ):
            h = model(batch)
            l_out = outcome_loss(model, h, batch) * CFG.lambda_outcome
            l_vote = vote_loss(model, h, batch) * vote_loss_weight()
            l_gate = gatekeep_loss(model, h, batch) * gatekeep_loss_weight()
            l_ctr = (
                contrastive_loss(model, h, batch, CFG.contrastive_tau)
                * CFG.lambda_contrast
            )
            l_tmp = temporal_smooth_loss(prev, h, keys) * CFG.lambda_temporal
            l_at = (
                actor_topic_loss_new(model, batch, h, model.topic_bank.get())
                * CFG.lambda_actor_topic
            )
            loss = (l_out + l_vote + l_gate + l_ctr + l_tmp + l_at) / ACCUM_STEPS
        loss.backward()
        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * ACCUM_STEPS
        steps += 1
        prev = {k: v.detach() for k, v in h.items()}
        del h, batch
        gc.collect()
        empty_cache_mps()
    return total / max(1, steps)


@torch.no_grad()
def infer_actor_topic_stance(model, data, emb, topic_protos, topk=10):
    te = F.normalize(topic_protos, dim=-1)
    outs = []
    for at in CFG.actor_types:
        if at not in emb:
            continue
        A = torch.from_numpy(emb[at]).to(DEVICE).float()
        Z = A.unsqueeze(1) * te.unsqueeze(0)
        Z2 = Z.view(-1, Z.size(-1))
        s = torch.tanh(model.actor_topic.stance(Z2)).view(A.size(0), te.size(0))
        c = model.actor_topic.conf(Z2).view(A.size(0), te.size(0))

        top_idx = torch.topk(s, k=min(topk, s.size(1)), dim=1).indices
        rows = []
        for i in range(A.size(0)):
            sel = top_idx[i]
            for j in sel:
                rows.append(
                    [at, i, int(j), float(s[i, j].item()), float(c[i, j].item())]
                )
        outs.append(
            pd.DataFrame(
                rows,
                columns=["actor_type", "actor_idx", "topic", "stance", "stance_conf"],
            )
        )
    return (
        pd.concat(outs, ignore_index=True)
        if outs
        else pd.DataFrame(
            columns=["actor_type", "actor_idx", "topic", "stance", "stance_conf"]
        )
    )


@torch.no_grad()
def infer_influence(model, emb):
    outs = []
    for at in CFG.actor_types:
        if at in emb:
            x = torch.from_numpy(emb[at]).to(DEVICE).float()
            I = model.influence(x).cpu().numpy()
            outs.append(
                pd.DataFrame(
                    {
                        "actor_type": at,
                        "actor_idx": np.arange(x.size(0)),
                        "influence": I,
                    }
                )
            )
    return (
        pd.concat(outs, ignore_index=True)
        if outs
        else pd.DataFrame(columns=["actor_type", "actor_idx", "influence"])
    )


def build_actor_topic_outputs(final_graph, emb, save_prefix, topic_protos, model):
    df = infer_actor_topic_stance(model, final_graph, emb, topic_protos, topk=10)
    os.makedirs(CFG.save_dir, exist_ok=True)
    df.to_parquet(
        os.path.join(CFG.save_dir, f"{save_prefix}_actor_topic.parquet"), index=False
    )


def build_overall_influence_series(emb, save_prefix, model):
    df = infer_influence(model, emb)
    os.makedirs(CFG.save_dir, exist_ok=True)
    df.to_parquet(
        os.path.join(CFG.save_dir, f"{save_prefix}_overall_influence.parquet"),
        index=False,
    )


@torch.no_grad()
def calibrate_outcome_temperature(model, data):
    if "bill" not in data.node_types:
        return 1.0
    input_nodes = ("bill", torch.arange(data["bill"].num_nodes))
    loader = NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors={et: [-1] * CFG.layers for et in data.edge_types},
        batch_size=4096,
        shuffle=False,
    )
    logits = []
    ys = []
    model.eval()
    for b in loader:
        b = b.to(DEVICE)
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.bfloat16, enabled=True
        ):
            h = model(b)
            y = _node_attr(b, "bill", "y", h["bill"].size(0))
            if y is None:
                continue
            logits.append(model.outcome(h["bill"]))
            ys.append(y.to(DEVICE))
    if not logits:
        return 1.0
    logits = torch.cat(logits)
    y = torch.cat(ys).long()
    T = torch.tensor(1.0, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(T.detach().cpu())


def run_train(data_path, save_prefix="leginflu_v10"):
    os.makedirs(CFG.save_dir, exist_ok=True)
    data = load_hetero(data_path)
    ensure_bidirectional(data)
    normalize_node_features(data)
    pe = per_type_laplacian_pe(data)
    K = (
        int(torch.max(data["bill"].cluster).item()) + 1
        if "cluster" in data["bill"]
        else 1
    )
    slices = build_time_slices(data)
    if not slices:
        slices = [None]
    model = Model(data, CFG.d, CFG.layers, CFG.drop, pe, K).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    hist = []
    full_graph = filter_graph_by_time(
        data, slices[-1] if slices[0] is not None else None
    )
    with torch.no_grad():
        emb_full = infer_embeddings(model, full_graph, bsz=min(4096, CFG.bsz))
        protos = compute_topic_protos_from_full(full_graph, emb_full)
        model.topic_bank.protos.data = protos
    snap_ids = list(range(len(slices)))
    for sid in tqdm(snap_ids, desc="snapshots"):
        snap = filter_graph_by_time(
            data, slices[sid] if slices[0] is not None else None
        )
        for e in tqdm(range(CFG.epochs), desc=f"snapshot {sid} epochs", leave=False):
            loss = train_one_snapshot(model, snap, opt, sid, protos)
            hist.append({"snapshot": sid, "epoch": e, "loss": float(loss)})
    pd.DataFrame(hist).to_parquet(
        os.path.join(CFG.save_dir, f"{save_prefix}_train_hist.parquet"), index=False
    )
    final_graph = filter_graph_by_time(
        data, slices[-1] if slices[0] is not None else None
    )
    emb = infer_embeddings(model, final_graph, bsz=min(4096, CFG.bsz))
    topic_protos = compute_topic_protos_from_full(final_graph, emb)
    T = calibrate_outcome_temperature(model, final_graph)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": {
                k: getattr(CFG, k)
                for k in CFG.__dict__.keys()
                if not k.startswith("__")
            },
            "temp": T,
        },
        os.path.join(CFG.save_dir, f"{save_prefix}_model.pt"),
    )
    torch.save(emb, os.path.join(CFG.save_dir, f"{save_prefix}_embeddings.pt"))
    build_actor_topic_outputs(final_graph, emb, save_prefix, topic_protos, model)
    build_overall_influence_series(emb, save_prefix, model)
    return {
        "model": os.path.join(CFG.save_dir, f"{save_prefix}_model.pt"),
        "embeddings": os.path.join(CFG.save_dir, f"{save_prefix}_embeddings.pt"),
        "hist": os.path.join(CFG.save_dir, f"{save_prefix}_train_hist.parquet"),
    }


if __name__ == "__main__":
    outs = run_train("data5.pt", save_prefix="leginflu_v10")
    print(json.dumps(outs, indent=2))
