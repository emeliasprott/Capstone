import os, math, gc, warnings, random, torch, numpy as np, pandas as pd
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _dev():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _dev()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


@dataclass
class CFG:
    d = 160
    drop = 0.15
    layers = 2
    lr = 1e-3
    wd = 1e-4
    max_grad = 1.0
    vote_bsz = 4096
    gate_bsz = 2048
    bill_chunk = 8192
    ls = 0.05
    outcome_classes = 2
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    topics_expected = 77
    min_eff_votes = 5
    seed = 42


def nanfix(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def remap_vote_targets(y_raw):
    y = y_raw.long()
    y = torch.where(y == -1, torch.zeros_like(y), y)
    y = torch.where(y == 0, torch.ones_like(y), y)
    y = torch.where(y == 1, torch.full_like(y, 2), y)
    return y


def remap_outcome_targets(y_raw, k):
    y = y_raw.clone()
    if k == 2:
        y[y <= 0] = 0
        y[y == 1] = 1
    else:
        y[y == -1] = 0
        y[y == 0] = 1
        y[y == 1] = 2
    return y.long()


def balanced_ce(logits, target, num_classes, label_smoothing=0.0):
    m = target.ge(0)
    if not m.any():
        return logits.sum() * 0
    logits, target = logits[m], target[m]
    one = F.one_hot(target, num_classes).float()
    if label_smoothing > 0:
        one = (1 - label_smoothing) * one + label_smoothing / num_classes
    logp = F.log_softmax(logits, -1)
    cnt = (
        torch.stack([(target == c).sum() for c in range(num_classes)])
        .float()
        .clamp_min(1)
    )
    w = cnt.sum() / cnt
    w = w / w.mean()
    return (-(w[target] * (one * logp).sum(-1))).mean()


def brier(logits, target, C):
    m = target.ge(0)
    if not m.any():
        return logits.sum() * 0
    p = F.softmax(logits[m], -1)
    y = F.one_hot(target[m], C).float()
    return ((p - y) ** 2).mean()


def idx_on(x, like):
    return x.to(like.device).long()


def macro_f1(logits, target):
    m = target.ge(0)
    if not m.any():
        return torch.tensor(0.0, device=logits.device)
    pred = logits[m].argmax(-1)
    t = target[m]
    C = logits.size(-1)
    f = 0.0
    for c in range(C):
        tp = ((pred == c) & (t == c)).sum().float()
        fp = ((pred == c) & (t != c)).sum().float()
        fn = ((pred != c) & (t == c)).sum().float()
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f += 2 * p * r / (p + r + 1e-6)
    return f / C


def attach_topics(data, expected=None):
    if hasattr(data["bill"], "cluster"):
        raw = data["bill"].cluster.long()
    else:
        raise RuntimeError("bill.cluster required")
    mask = raw.ge(0)
    uniq = torch.unique(raw[mask], sorted=True)
    remap = {int(t): i for i, t in enumerate(uniq.tolist())}
    topic_ix = torch.full_like(raw, -1)
    topic_ix[mask] = torch.tensor([remap[int(v)] for v in raw[mask].tolist()])
    T = len(uniq)
    data["topic"].num_nodes = T
    src = topic_ix[mask]
    dst = torch.nonzero(mask, as_tuple=False).view(-1)
    data[("topic", "has", "bill")].edge_index = torch.stack([src, dst], 0)
    data[("topic", "has", "bill")].edge_attr = torch.ones(dst.numel(), 1)
    counts = torch.bincount(topic_ix[topic_ix >= 0], minlength=T).float()
    data["topic"].prev = (counts / counts.sum().clamp_min(1)).to(torch.float32)
    data["bill"].topic_ix = topic_ix
    if expected is not None and T != expected:
        print(f"[topic]={T}")
    return T, topic_ix


def mlp(dims, drop=0.0):
    m = []
    for i in range(len(dims) - 1):
        m.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            m.append(nn.ReLU())
            if drop > 0:
                m.append(nn.Dropout(drop))
    return nn.Sequential(*m)


class Projector(nn.Module):
    def __init__(self, in_dims, d):
        super().__init__()
        self.map = nn.ModuleDict()
        for nt, dim in in_dims.items():
            if dim <= 0:
                continue
            if nt in ("legislator_term", "committee", "donor"):
                hid = max(64, min(128, dim))
                self.map[nt] = mlp([dim, hid, d], drop=0.0)
            else:
                self.map[nt] = nn.Linear(dim, d)

    def forward(self, xd):
        out = {}
        for nt, x in xd.items():
            if x is None:
                continue
            out[nt] = self.map[nt](nanfix(x))
        return out


class HetBlock(nn.Module):
    def __init__(self, metadata, d, drop=0.15):
        super().__init__()
        nts, ets = metadata
        self.nts = list(nts)
        self.ets = list(ets)
        self.lin = nn.ModuleDict(
            {"__".join(et): nn.Linear(2 * d, d) for et in self.ets}
        )
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(d) for nt in self.nts})
        self.drop = nn.Dropout(drop)

    def forward(self, h, batch):
        out = {nt: h[nt].new_zeros(h[nt].size(0), h[nt].size(1)) for nt in h}
        for et in self.ets:
            if et[0] not in h or et[2] not in h:
                continue
            store = batch[et]
            if store.edge_index.numel() == 0:
                continue
            s, d = store.edge_index
            xs = h[et[0]]
            xd = h[et[2]]
            agg = scatter_mean(
                xs.index_select(0, s.long()), d.long(), dim=0, dim_size=xd.size(0)
            )
            y = self.lin["__".join(et)](torch.cat([xd, agg], -1))
            out[et[2]] += y
        for nt in out:
            out[nt] = self.drop(self.norm[nt](out[nt]))
        return out


class Backbone(nn.Module):
    def __init__(self, metadata, d, layers, drop):
        super().__init__()
        self.blocks = nn.ModuleList(
            [HetBlock(metadata, d, drop) for _ in range(layers)]
        )

    def forward(self, h, batch):
        for b in self.blocks:
            h = b(h, batch)
        return h


class VoteHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = mlp([3 * d, d], drop=0.1)
        self.out = nn.Linear(d, 3)

    def forward(self, lt, bill, topic):
        return self.out(self.net(torch.cat([lt, bill, topic], -1)))


class OutcomeHead(nn.Module):
    def __init__(self, d, C=2):
        super().__init__()
        self.net = mlp([3 * d + 1, d], drop=0.1)
        self.out = nn.Linear(d, C)

    def forward(self, bill, comm_ctx, topic, margin1):
        return self.out(self.net(torch.cat([bill, comm_ctx, topic, margin1], -1)))


class GateHead(nn.Module):
    def __init__(self, d, stages=6):
        super().__init__()
        self.net = mlp([3 * d, d], drop=0.1)
        self.out = nn.Linear(d, stages)

    def forward(self, committee, bill, topic):
        return self.out(self.net(torch.cat([committee, bill, topic], -1)))


def build_in_dims(data):
    d = {}
    for nt in data.node_types:
        if nt == "topic":
            continue
        x = getattr(data[nt], "x", None)
        d[nt] = x.size(-1) if x is not None else 0
    return d


def build_version_to_bill(data):
    eb = data[("bill_version", "is_version", "bill")].edge_index
    m = torch.zeros(data["bill_version"].num_nodes, dtype=torch.long)
    m[eb[0]] = eb[1]
    return m


def split_indices(N, cfg, seed):
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_train = int(N * cfg.train_ratio)
    n_val = int(N * cfg.val_ratio)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def cover_all_neighbors(data, hops=2, val=-1):
    return {et: [val] * hops for et in data.edge_types}


def cover_with_overrides(data, overrides, hops=2, default=0):
    caps = {et: [default] * hops for et in data.edge_types}
    for et, v in overrides.items():
        if et in caps:
            caps[et] = v
    return caps


def build_neighbor_caps(data):
    caps = {}
    caps[("legislator_term", "voted_on", "bill_version")] = [48, 24]
    caps[("bill_version", "is_version", "bill")] = [2, 1]
    caps[("bill_version", "priorVersion", "bill_version")] = [2, 1]
    caps[("bill_version", "read", "committee")] = [8, 4]
    caps[("legislator_term", "member_of", "committee")] = [8, 4]
    caps[("legislator_term", "wrote", "bill_version")] = [8, 4]
    caps[("donor", "donated_to", "legislator_term")] = [8, 4]
    caps[("lobby_firm", "lobbied", "legislator_term")] = [8, 4]
    caps[("lobby_firm", "lobbied", "committee")] = [6, 3]
    caps[("topic", "has", "bill")] = [8, 4]
    return caps


def make_num_neighbors(data, over):
    out = {("*", "*", "*"): [0, 0]}
    for et in data.edge_types:
        out[et] = over.get(et, [0, 0])
        rev = (et[2], "rev_" + et[1], et[0])
        if rev in data.edge_types:
            out[rev] = over.get(et, [0, 0])
    return out


class Trainer:
    def __init__(self, data, cfg):
        self.cfg = cfg
        base = data.clone()
        base = ToUndirected()(base)
        base = RemoveIsolatedNodes()(base)
        for et in base.edge_types:
            base[et].edge_index = base[et].edge_index.long()
        self.data = base
        self.T, self.topic_ix = attach_topics(self.data, cfg.topics_expected)
        in_dims = build_in_dims(self.data)
        self.proj = Projector(in_dims, cfg.d).to(DEVICE)
        self.backbone = Backbone(self.data.metadata(), cfg.d, cfg.layers, cfg.drop).to(
            DEVICE
        )
        self.vote_head = VoteHead(cfg.d).to(DEVICE)
        self.out_head = OutcomeHead(cfg.d, cfg.outcome_classes).to(DEVICE)
        self.gate_head = GateHead(cfg.d).to(DEVICE)
        self.opt = torch.optim.AdamW(
            list(self.proj.parameters())
            + list(self.backbone.parameters())
            + list(self.vote_head.parameters())
            + list(self.out_head.parameters())
            + list(self.gate_head.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.wd,
        )
        self.bv2b = build_version_to_bill(self.data)
        self.topic_emb = nn.Embedding(self.T, cfg.d).to(DEVICE)
        lt_train, lt_val, lt_test = split_indices(
            self.data["legislator_term"].num_nodes, cfg, cfg.seed + 1
        )
        bill_train, bill_val, bill_test = split_indices(
            self.data["bill"].num_nodes, cfg, cfg.seed + 2
        )
        com_train, com_val, com_test = split_indices(
            self.data["committee"].num_nodes, cfg, cfg.seed + 3
        )
        self.split = {
            "lt": {"train": lt_train, "val": lt_val, "test": lt_test},
            "bill": {"train": bill_train, "val": bill_val, "test": bill_test},
            "committee": {"train": com_train, "val": com_val, "test": com_test},
        }
        self.vote_loader_train = self._build_vote_loader(
            self.split["lt"]["train"], shuffle=True
        )
        self.vote_loader_val = self._build_vote_loader(
            self.split["lt"]["val"], shuffle=False
        )
        self.gate_loader_train = self._build_gate_loader(
            self.split["committee"]["train"], shuffle=True
        )
        self.gate_loader_val = self._build_gate_loader(
            self.split["committee"]["val"], shuffle=False
        )
        self.bill_ctx = None
        self.cached_bill_emb = None

    def _build_vote_loader(self, lt_nodes, shuffle):
        over = {}
        over[("legislator_term", "voted_on", "bill_version")] = [96, 48]
        over[("bill_version", "is_version", "bill")] = [1, 0]
        over[("bill_version", "priorVersion", "bill_version")] = [1, 1]
        over[("bill_version", "read", "committee")] = [16, 8]
        over[("legislator_term", "member_of", "committee")] = [16, 8]
        over[("legislator_term", "wrote", "bill_version")] = [16, 8]
        over[("donor", "donated_to", "legislator_term")] = [8, 4]
        over[("lobby_firm", "lobbied", "legislator_term")] = [8, 4]
        over[("lobby_firm", "lobbied", "committee")] = [6, 3]
        over[("topic", "has", "bill")] = [96, 48]
        caps = cover_with_overrides(self.data, over, hops=2, default=0)
        return NeighborLoader(
            self.data,
            input_nodes=("legislator_term", lt_nodes),
            num_neighbors=caps,
            batch_size=self.cfg.vote_bsz,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    def _build_gate_loader(self, committee_nodes, shuffle):
        over = {}
        over[("bill_version", "read", "committee")] = [16, 8]
        over[("bill_version", "is_version", "bill")] = [1, 0]
        over[("topic", "has", "bill")] = [96, 48]
        over[("legislator_term", "member_of", "committee")] = [8, 4]
        caps = cover_with_overrides(self.data, over, hops=2, default=0)
        return NeighborLoader(
            self.data,
            input_nodes=("committee", committee_nodes),
            num_neighbors=caps,
            batch_size=self.cfg.gate_bsz,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    def encode_batch(self, batch):
        xd = {}
        for nt in batch.node_types:
            if nt == "topic":
                continue
            x = getattr(batch[nt], "x", None)
            xd[nt] = x.to(DEVICE) if x is not None else None
        h = self.proj(xd)
        for nt in batch.node_types:
            if nt not in h and nt != "topic":
                h[nt] = torch.zeros(batch[nt].num_nodes, self.cfg.d, device=DEVICE)
        h = self.backbone(h, batch.to(DEVICE))
        h["topic"] = self.topic_emb.weight
        return h

    def precompute_static(self):
        caps = cover_all_neighbors(self.data, hops=2, val=-1)
        loader = NeighborLoader(
            self.data,
            input_nodes=("bill", torch.arange(self.data["bill"].num_nodes)),
            num_neighbors=caps,
            batch_size=self.cfg.bill_chunk,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        all_bill = torch.zeros(self.data["bill"].num_nodes, self.cfg.d)
        all_committee = torch.zeros(self.data["bill"].num_nodes, self.cfg.d)
        for b in loader:
            h = self.encode_batch(b)
            all_bill[b["bill"].n_id.cpu()] = h["bill"].detach().cpu()

            if ("bill_version", "read", "committee") in b.edge_types:
                r_src, r_dst = b[("bill_version", "read", "committee")].edge_index
                r_src = idx_on(r_src, h["bill_version"])
                r_dst = idx_on(r_dst, h["committee"])
                bv_pool = scatter_mean(
                    h["committee"].index_select(0, r_dst),
                    r_src,
                    dim=0,
                    dim_size=h["bill_version"].size(0),
                )
                if ("bill_version", "is_version", "bill") in b.edge_types:
                    eb = b[("bill_version", "is_version", "bill")].edge_index
                    eb0 = idx_on(eb[0], h["bill_version"])
                    eb1 = idx_on(eb[1], h["bill"])
                    comm_bill = scatter_mean(
                        bv_pool.index_select(0, eb0),
                        eb1,
                        dim=0,
                        dim_size=h["bill"].size(0),
                    )
                    all_committee[b["bill"].n_id.cpu()] = comm_bill.detach().cpu()
            del b, h
            gc.collect()

        self.bill_ctx = all_committee
        self.cached_bill_emb = all_bill

    def vote_pass(self, loader):
        stats = {"loss_vote": 0.0}
        bill_margin = torch.zeros(self.data["bill"].num_nodes, device=DEVICE)
        nsteps = 0
        et = ("legislator_term", "voted_on", "bill_version")
        for batch in loader:
            h = self.encode_batch(batch)
            et = ("legislator_term", "voted_on", "bill_version")
            if et not in batch.edge_types or batch[et].edge_index.numel() == 0:
                del batch, h
                continue
            lt_i, bv_i = batch[et].edge_index

            if ("bill_version", "is_version", "bill") in batch.edge_types:
                eb = batch[("bill_version", "is_version", "bill")].edge_index
                dev = eb.device
                b_of = torch.full(
                    (batch["bill_version"].num_nodes,), -1, dtype=torch.long, device=dev
                )
                b_of[eb[0].long()] = eb[1].long()
                b_local = b_of[bv_i.to(dev)]
            else:
                del batch, h
                continue

            m = b_local.ge(0)
            if not m.any():
                del batch, h
                continue

            lt_idx = idx_on(lt_i[m], h["legislator_term"])
            b_idx = idx_on(b_local[m], h["bill"])

            lt_h = h["legislator_term"].index_select(0, lt_idx)
            bill_h = h["bill"].index_select(0, b_idx)

            t_idx_glob = self.topic_ix[batch["bill"].n_id[b_idx.cpu()]].clamp(min=0)
            topic_h = h["topic"].index_select(0, t_idx_glob.to(h["topic"].device))

            raw = (
                batch[et].edge_attr[:, 0]
                if getattr(batch[et], "edge_attr", None) is not None
                else torch.full((lt_i.size(0),), -1000)
            )
            target = remap_vote_targets(raw[m].to(h["bill"].device))

            logits = self.vote_head(lt_h, bill_h, topic_h)
            target = remap_vote_targets(raw[m].to(DEVICE))
            loss = balanced_ce(logits, target, 3, self.cfg.ls) + 0.1 * brier(
                logits, target, 3
            )
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.proj.parameters())
                + list(self.backbone.parameters())
                + list(self.vote_head.parameters()),
                self.cfg.max_grad,
            )
            self.opt.step()
            with torch.no_grad():
                p = F.softmax(logits, -1)
                contrib = p[:, 2] - p[:, 0]
                b_glob = batch["bill"].n_id[b_local[m]].cpu()
                tmp = torch.zeros(self.data["bill"].num_nodes, device=DEVICE)
                tmp.index_add_(0, b_glob.to(DEVICE), contrib)
                bill_margin += tmp
                stats["loss_vote"] += float(loss.item())
                nsteps += 1
            del batch, h, lt_h, bill_h, topic_h, logits, target
            gc.collect()
        if nsteps > 0:
            stats["loss_vote"] /= nsteps
        return stats, bill_margin.detach()

    def outcome_pass(self, bill_ids, bill_margin):
        stats = {"loss_out": 0.0, "out_f1": 0.0}
        N = bill_ids.numel()
        for s in range(0, N, self.cfg.bill_chunk):
            idx = bill_ids[s : s + self.cfg.bill_chunk]
            bill_h = self.cached_bill_emb[idx.cpu()].to(DEVICE)
            topic_h = self.topic_emb.weight.index_select(
                0, self.topic_ix[idx.cpu()].clamp(min=0).to(DEVICE)
            )
            comm_ctx = (
                self.bill_ctx[idx.cpu()].to(DEVICE)
                if self.bill_ctx is not None
                else torch.zeros_like(bill_h)
            )
            margin = bill_margin[idx.to(DEVICE)].view(-1, 1)
            logits = self.out_head(bill_h, comm_ctx, topic_h, margin)
            tgt = getattr(self.data["bill"], "y", None)
            if tgt is None:
                del bill_h, topic_h, comm_ctx, margin, logits
                gc.collect()
                continue
            labels = remap_outcome_targets(
                tgt[idx].to(DEVICE), self.cfg.outcome_classes
            )
            loss = balanced_ce(
                logits, labels, self.cfg.outcome_classes, self.cfg.ls
            ) + 0.1 * brier(logits, labels, self.cfg.outcome_classes)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.out_head.parameters()), self.cfg.max_grad
            )
            self.opt.step()
            with torch.no_grad():
                stats["loss_out"] += float(loss.item())
                stats["out_f1"] += float(macro_f1(logits, labels).item())
            del bill_h, topic_h, comm_ctx, margin, logits, labels
            gc.collect()
        denom = max((N + self.cfg.bill_chunk - 1) // self.cfg.bill_chunk, 1)
        stats["loss_out"] /= denom
        stats["out_f1"] /= denom
        return stats

    def gate_pass(self, loader):
        stats = {"loss_gate": 0.0}
        n = 0
        et = ("bill_version", "read", "committee")
        for batch in loader:
            h = self.encode_batch(batch)
            et = ("bill_version", "read", "committee")
            if et not in batch.edge_types or batch[et].edge_index.numel() == 0:
                del batch, h
                continue
            bv_i, c_i = batch[et].edge_index

            if ("bill_version", "is_version", "bill") in batch.edge_types:
                eb = batch[("bill_version", "is_version", "bill")].edge_index
                dev = eb.device
                b_of = torch.full(
                    (batch["bill_version"].num_nodes,), -1, dtype=torch.long, device=dev
                )
                b_of[eb[0].long()] = eb[1].long()
                b_local = b_of[bv_i.to(dev)]
            else:
                del batch, h
                continue

            m = b_local.ge(0)
            if not m.any():
                del batch, h
                continue

            c_idx = idx_on(c_i[m], h["committee"])
            b_idx = idx_on(b_local[m], h["bill"])

            com_h = h["committee"].index_select(0, c_idx)
            bill_h = h["bill"].index_select(0, b_idx)

            t_idx_glob = self.topic_ix[batch["bill"].n_id[b_idx.cpu()]].clamp(min=0)
            topic_h = h["topic"].index_select(0, t_idx_glob.to(h["topic"].device))

            logits = self.gate_head(com_h, bill_h, topic_h)
            if getattr(batch[et], "edge_attr", None) is not None and batch[
                et
            ].edge_attr.size(0) == bv_i.size(0):
                tgt = (
                    batch[et]
                    .edge_attr[m, 0]
                    .long()
                    .clamp(0, logits.size(-1) - 1)
                    .to(DEVICE)
                )
            else:
                tgt = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
            loss = balanced_ce(logits, tgt, logits.size(-1), self.cfg.ls)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.gate_head.parameters()), self.cfg.max_grad
            )
            self.opt.step()
            stats["loss_gate"] += float(loss.item())
            n += 1
            del batch, h, logits, tgt
            gc.collect()
        if n > 0:
            stats["loss_gate"] /= n
        return stats

    def stance_legislator_term(self):
        e = self.data[("legislator_term", "voted_on", "bill_version")]
        y = e.edge_attr[:, 0].float()
        bill = self.bv2b[e.edge_index[1]]
        topic = self.topic_ix[bill]
        m = topic.ge(0) & y.ne(0)
        lt = e.edge_index[0][m]
        t = topic[m]
        v = y[m]
        Nlt = self.data["legislator_term"].num_nodes
        T = int(self.topic_ix.max().item() + 1)
        lin = lt * T + t
        num = scatter_add(v, lin, dim=0, dim_size=Nlt * T).view(Nlt, T)
        den = scatter_add(torch.ones_like(v), lin, dim=0, dim_size=Nlt * T).view(Nlt, T)
        stance = torch.zeros_like(num)
        m2 = den > 0
        stance[m2] = num[m2] / (den[m2] + 10.0)
        stance[den < self.cfg.min_eff_votes] = float("nan")
        return stance

    def influence_inference(self):
        et = ("legislator_term", "voted_on", "bill_version")
        e = self.data[et]
        caps = make_num_neighbors(self.data, build_neighbor_caps(self.data))
        lt_all = []
        bill_all = []
        topic_all = []
        yes_all = []
        no_all = []
        src = e.edge_index[0]
        idx_all = torch.arange(self.data["legislator_term"].num_nodes)
        loader = NeighborLoader(
            self.data,
            input_nodes=("legislator_term", idx_all),
            num_neighbors=caps,
            batch_size=self.cfg.vote_bsz,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        for batch in loader:
            h = self.encode_batch(batch)
            et = ("legislator_term", "voted_on", "bill_version")
            if et not in batch.edge_types or batch[et].edge_index.numel() == 0:
                del batch, h
                continue
            lt_i, bv_i = batch[et].edge_index

            if ("bill_version", "is_version", "bill") in batch.edge_types:
                eb = batch[("bill_version", "is_version", "bill")].edge_index
                dev = eb.device
                b_of = torch.full(
                    (batch["bill_version"].num_nodes,), -1, dtype=torch.long, device=dev
                )
                b_of[eb[0].long()] = eb[1].long()
                b_local = b_of[bv_i.to(dev)]
            else:
                del batch, h
                continue

            m = b_local.ge(0)
            if not m.any():
                del batch, h
                continue

            lt_idx = idx_on(lt_i[m], h["legislator_term"])
            b_idx = idx_on(b_local[m], h["bill"])

            lt_h = h["legislator_term"].index_select(0, lt_idx)
            bill_h = h["bill"].index_select(0, b_idx)

            t_idx_glob = self.topic_ix[batch["bill"].n_id[b_idx.cpu()]].clamp(min=0)
            topic_h = h["topic"].index_select(0, t_idx_glob.to(h["topic"].device))
            logits = self.vote_head(lt_h, bill_h, topic_h)
            p = F.softmax(logits, -1).detach().cpu()
            yes_all.append(p[:, 2])
            no_all.append(p[:, 0])
            lt_all.append(batch["legislator_term"].n_id[lt_i[m]].cpu())
            gl_bv = batch["bill_version"].n_id[bv_i[m]].cpu()
            bill_all.append(self.bv2b[gl_bv])
            topic_all.append(self.topic_ix[self.bv2b[gl_bv]])
            del batch, h, lt_h, bill_h, topic_h, logits, p
            gc.collect()
        if len(lt_all) == 0:
            return {
                "edges": None,
                "bill_margin": torch.zeros(self.data["bill"].num_nodes),
            }
        lt_all = torch.cat(lt_all)
        bill_all = torch.cat(bill_all)
        topic_all = torch.cat(topic_all)
        p_yes = torch.cat(yes_all)
        p_no = torch.cat(no_all)
        bill_margin = torch.zeros(self.data["bill"].num_nodes)
        bill_margin.index_add_(0, bill_all, (p_yes - p_no))
        return {
            "edges": {
                "lt": lt_all,
                "bill": bill_all,
                "topic": topic_all,
                "p_yes": p_yes,
                "p_no": p_no,
            },
            "bill_margin": bill_margin,
        }

    def build_outputs(self, inf):
        T = self.T
        topic_prev = (
            self.data["topic"].prev
            if hasattr(self.data["topic"], "prev")
            else torch.ones(T) / max(T, 1)
        )
        N_bill = self.data["bill"].num_nodes
        N_lt = self.data["legislator_term"].num_nodes
        N_c = self.data["committee"].num_nodes
        N_dn = self.data["donor"].num_nodes if "donor" in self.data.node_types else 0
        N_lb = (
            self.data["lobby_firm"].num_nodes
            if "lobby_firm" in self.data.node_types
            else 0
        )

        lt_topic_infl = torch.zeros(N_lt, T)
        lt_topic_eng = torch.zeros(N_lt, T)
        if inf["edges"] is not None:
            lt_glob = inf["edges"]["lt"]
            bill_glob = inf["edges"]["bill"]
            topic_glob = inf["edges"]["topic"]
            contrib = (inf["edges"]["p_yes"] - inf["edges"]["p_no"]).float()
            m = topic_glob >= 0
            lin = lt_glob[m] * T + topic_glob[m]
            lt_topic_infl.view(-1).index_add_(0, lin, contrib[m])
            lt_topic_eng.view(-1).index_add_(0, lin, torch.ones_like(contrib[m]))

        dn_topic_infl = torch.zeros(N_dn, T)
        dn_topic_eng = torch.zeros(N_dn, T)
        if (
            "donor",
            "donated_to",
            "legislator_term",
        ) in self.data.edge_types and N_dn > 0:
            dn = self.data[("donor", "donated_to", "legislator_term")].edge_index
            dn_src, dn_dst = dn
            deg = torch.bincount(dn_dst, minlength=N_lt).clamp_min(1)
            nz = lt_topic_eng.nonzero(as_tuple=False)
            if nz.numel() > 0:
                order = torch.argsort(dn_dst)
                dn_src_sorted = dn_src[order]
                ptr = torch.zeros(N_lt + 1, dtype=torch.long)
                cnt = torch.bincount(dn_dst, minlength=N_lt)
                ptr[1:] = torch.cumsum(cnt, 0)
                for i in range(nz.size(0)):
                    a, t = int(nz[i, 0]), int(nz[i, 1])
                    if cnt[a] == 0:
                        continue
                    s, e = int(ptr[a]), int(ptr[a + 1])
                    ids = dn_src_sorted[s:e]
                    dn_topic_eng[ids, t] += float(1.0 / max(1, int(deg[a].item())))
            nz = lt_topic_infl.nonzero(as_tuple=False)
            if nz.numel() > 0:
                order = torch.argsort(dn_dst)
                dn_src_sorted = dn_src[order]
                ptr = torch.zeros(N_lt + 1, dtype=torch.long)
                cnt = torch.bincount(dn_dst, minlength=N_lt)
                ptr[1:] = torch.cumsum(cnt, 0)
                for i in range(nz.size(0)):
                    a, t = int(nz[i, 0]), int(nz[i, 1])
                    if cnt[a] == 0:
                        continue
                    s, e = int(ptr[a]), int(ptr[a + 1])
                    ids = dn_src_sorted[s:e]
                    v = float(lt_topic_infl[a, t].item()) / max(1, int(deg[a].item()))
                    dn_topic_infl[ids, t] += v

        lb_topic_infl = torch.zeros(N_lb, T)
        lb_topic_eng = torch.zeros(N_lb, T)
        if (
            "lobby_firm",
            "lobbied",
            "legislator_term",
        ) in self.data.edge_types and N_lb > 0:
            lb = self.data[("lobby_firm", "lobbied", "legislator_term")].edge_index
            lb_src, lb_dst = lb
            deg = torch.bincount(lb_dst, minlength=N_lt).clamp_min(1)
            nz = lt_topic_eng.nonzero(as_tuple=False)
            if nz.numel() > 0:
                order = torch.argsort(lb_dst)
                lb_src_sorted = lb_src[order]
                ptr = torch.zeros(N_lt + 1, dtype=torch.long)
                cnt = torch.bincount(lb_dst, minlength=N_lt)
                ptr[1:] = torch.cumsum(cnt, 0)
                for i in range(nz.size(0)):
                    a, t = int(nz[i, 0]), int(nz[i, 1])
                    if cnt[a] == 0:
                        continue
                    s, e = int(ptr[a]), int(ptr[a + 1])
                    ids = lb_src_sorted[s:e]
                    lb_topic_eng[ids, t] += float(1.0 / max(1, int(deg[a].item())))
            nz = lt_topic_infl.nonzero(as_tuple=False)
            if nz.numel() > 0:
                order = torch.argsort(lb_dst)
                lb_src_sorted = lb_src[order]
                ptr = torch.zeros(N_lt + 1, dtype=torch.long)
                cnt = torch.bincount(lb_dst, minlength=N_lt)
                ptr[1:] = torch.cumsum(cnt, 0)
                for i in range(nz.size(0)):
                    a, t = int(nz[i, 0]), int(nz[i, 1])
                    if cnt[a] == 0:
                        continue
                    s, e = int(ptr[a]), int(ptr[a + 1])
                    ids = lb_src_sorted[s:e]
                    v = float(lt_topic_infl[a, t].item()) / max(1, int(deg[a].item()))
                    lb_topic_infl[ids, t] += v

        if ("bill_version", "read", "committee") in self.data.edge_types and (
            "bill_version",
            "is_version",
            "bill",
        ) in self.data.edge_types:
            rd = self.data[("bill_version", "read", "committee")].edge_index
            bv, cm = rd
            b = self.bv2b[bv]
            t = self.topic_ix[b].clamp(min=0)
            c_topic_eng = torch.zeros(N_c, T)
            c_topic_infl = torch.zeros(N_c, T)
            c_topic_eng.view(-1).index_add_(0, (cm * T + t), torch.ones_like(b).float())
            b_deg = torch.bincount(b, minlength=N_bill).clamp_min(1)
            share = (1.0 / b_deg[b]).float()
            add = inf["bill_margin"][b.cpu()].to(torch.float32) * share.cpu()
            c_topic_infl.view(-1).index_add_(0, (cm * T + t), add)
        else:
            c_topic_eng = torch.zeros(N_c, T)
            c_topic_infl = torch.zeros(N_c, T)

        lt_stance = self.stance_legislator_term()

        def aligned_stance(num_src, edge_index, dst_stance):
            if num_src == 0:
                return torch.zeros(0, dst_stance.size(1))
            if edge_index is None or edge_index.numel() == 0 or dst_stance.numel() == 0:
                return torch.zeros(num_src, dst_stance.size(1))
            s, d = edge_index
            S_num = torch.zeros(num_src, dst_stance.size(1))
            S_den = torch.zeros_like(S_num)
            val = torch.where(
                torch.isfinite(dst_stance[d]),
                dst_stance[d],
                torch.zeros_like(dst_stance[d]),
            )
            S_num.index_add_(0, s, val)
            S_den.index_add_(
                0,
                s,
                torch.where(
                    torch.isfinite(dst_stance[d]),
                    torch.ones_like(dst_stance[d]),
                    torch.zeros_like(dst_stance[d]),
                ),
            )
            out = torch.zeros_like(S_num)
            m = S_den > 0
            out[m] = S_num[m] / (S_den[m] + 1e-6)
            return out

        dn_stance = aligned_stance(
            N_dn,
            (
                self.data[("donor", "donated_to", "legislator_term")].edge_index
                if ("donor", "donated_to", "legislator_term") in self.data.edge_types
                else None
            ),
            lt_stance,
        )
        lb_stance = aligned_stance(
            N_lb,
            (
                self.data[("lobby_firm", "lobbied", "legislator_term")].edge_index
                if ("lobby_firm", "lobbied", "legislator_term") in self.data.edge_types
                else None
            ),
            lt_stance,
        )
        c_stance = aligned_stance(
            N_c,
            (
                self.data[
                    ("legislator_term", "member_of", "committee")
                ].edge_index.flip(0)
                if ("legislator_term", "member_of", "committee") in self.data.edge_types
                else None
            ),
            lt_stance,
        )

        def rows_from(A, typ, infl, eng, stance):
            rows = []
            if A == 0:
                return rows
            eng_sum = eng.sum(1, keepdim=True).clamp_min(1.0)
            share = eng / eng_sum
            for a in range(A):
                for t in range(T):
                    s = (
                        float(stance[a, t].item())
                        if (stance.numel() > 0 and torch.isfinite(stance[a, t]))
                        else 0.0
                    )
                    i = float(infl[a, t].item())
                    e = float(eng[a, t].item())
                    sh = float(share[a, t].item())
                    ci = max(0.05, 0.2 / math.sqrt(1.0 + e))
                    rows.append(
                        {
                            "actor_id": a,
                            "actor_type": typ,
                            "topic_id": t,
                            "stance": s,
                            "stance_ci_lo": s - ci,
                            "stance_ci_hi": s + ci,
                            "influence_delta_mean": i,
                            "influence_ci_lo": i - ci,
                            "influence_ci_hi": i + ci,
                            "engagement": e,
                            "certainty": min(1.0, e / max(1.0, eng_sum[a, 0].item())),
                            "topic_share": sh,
                        }
                    )
            return rows

        rows = []
        rows += rows_from(
            N_lt, "legislator_term", lt_topic_infl, lt_topic_eng, lt_stance
        )
        rows += rows_from(N_dn, "donor", dn_topic_infl, dn_topic_eng, dn_stance)
        rows += rows_from(N_lb, "lobby_firm", lb_topic_infl, lb_topic_eng, lb_stance)
        rows += rows_from(N_c, "committee", c_topic_infl, c_topic_eng, c_stance)

        actor_topic_df = (
            pd.DataFrame(rows)
            .sort_values(["actor_type", "actor_id", "topic_id"])
            .reset_index(drop=True)
        )

        tp = topic_prev.numpy().astype(float)

        def overall_from(infl, eng, typ):
            out = []
            A = infl.shape[0] if infl.numel() > 0 else 0
            for a in range(A):
                e = eng[a].numpy().astype(float)
                W = e.sum()
                ww = (e / W) if W > 0 else np.full(T, 1.0 / max(1, T))
                val = float((ww * tp * infl[a].numpy().astype(float)).sum())
                out.append(
                    {
                        "actor_id": a,
                        "actor_type": typ,
                        "overall_influence": val,
                        "ci_lo": val - 0.1,
                        "ci_hi": val + 0.1,
                    }
                )
            return out

        actor_overall = []
        actor_overall += overall_from(lt_topic_infl, lt_topic_eng, "legislator_term")
        actor_overall += overall_from(dn_topic_infl, dn_topic_eng, "donor")
        actor_overall += overall_from(lb_topic_infl, lb_topic_eng, "lobby_firm")
        actor_overall += overall_from(c_topic_infl, c_topic_eng, "committee")

        actor_overall_df = (
            pd.DataFrame(actor_overall)
            .sort_values(["actor_type", "actor_id"])
            .reset_index(drop=True)
        )

        per_bill = [
            {
                "bill_id": int(i),
                "expected_margin": float(inf["bill_margin"][i].item()),
                "pivotal_actors": [],
                "committee_bottlenecks": [],
            }
            for i in range(N_bill)
        ]
        if inf["edges"] is not None:
            lt_glob = inf["edges"]["lt"]
            bill_glob = inf["edges"]["bill"]
            contrib = (inf["edges"]["p_yes"] - inf["edges"]["p_no"]).float()
            by = {}
            for i in range(bill_glob.numel()):
                b = int(bill_glob[i])
                a = int(lt_glob[i])
                v = float(contrib[i])
                if b not in by:
                    by[b] = {a: v}
                else:
                    by[b][a] = by[b].get(a, 0.0) + v
            for b, d in by.items():
                top = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                per_bill[b]["pivotal_actors"] = [
                    {"actor_id": aid, "score": float(s)} for aid, s in top
                ]
        per_bill_df = (
            pd.DataFrame(per_bill).sort_values("bill_id").reset_index(drop=True)
        )
        return {
            "per_bill": per_bill_df,
            "actor_topic": actor_topic_df,
            "actor_overall": actor_overall_df,
        }

    def train(self, epochs=2):
        self.precompute_static()
        for ep in range(epochs):
            s1, bill_margin = self.vote_pass(self.vote_loader_train)
            bill_ids = self.split["bill"]["train"]
            s2 = self.outcome_pass(bill_ids.to(torch.long), bill_margin.detach())
            s3 = self.gate_pass(self.gate_loader_train)
            v_loss = s1.get("loss_vote", float("nan"))
            o_loss = s2.get("loss_out", float("nan"))
            o_f1 = s2.get("out_f1", float("nan"))
            g_loss = s3.get("loss_gate", float("nan"))
            print(
                f"Epoch {ep+1}/{epochs}: vote_loss={v_loss:.4f} outcome_loss={o_loss:.4f} outcome_f1={o_f1:.4f} gate_loss={g_loss:.4f}"
            )

        return


def run(graph_path, epochs=2, save_prefix="legnn_eff"):
    data = torch.load(graph_path, weights_only=False)
    tr = Trainer(data, CFG())
    tr.train(epochs=epochs)
    inf = tr.influence_inference()
    outs = tr.build_outputs(inf)
    torch.save(
        {
            "model_proj": tr.proj.state_dict(),
            "model_backbone": tr.backbone.state_dict(),
            "vote_head": tr.vote_head.state_dict(),
            "out_head": tr.out_head.state_dict(),
            "gate_head": tr.gate_head.state_dict(),
            "topic_emb": tr.topic_emb.state_dict(),
            "cfg": tr.cfg.__dict__,
        },
        f"{save_prefix}_state.pt",
    )
    torch.save(outs, f"{save_prefix}_outputs.pt")
    return outs


if __name__ == "__main__":
    try:
        outs = run("data5.pt", epochs=15, save_prefix="legnn_eff")
    except Exception as e:
        print(f"Error during run: {e}")
