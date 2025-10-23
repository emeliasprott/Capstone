import os, math, random, gc, warnings, torch
import numpy as np
from tqdm import tqdm
from torch import nn
import pandas as pd
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_scatter import scatter_add, scatter_mean
from dataclasses import dataclass

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _device()
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ------------------------------- Config --------------------------------------
@dataclass
class CFG:
    d=128; drop=0.15; heads=4; layers=2
    lr=2e-3; wd=1e-4
    vote_bsz=88; bill_bsz=88; gate_bsz=88; eval_bsz=124
    time2vec_k=8
    topics_expected=437
    tau=0.7; ls=0.05; ece_bins=15
    alpha=1.0; beta=1.0; gamma=0.5; delta=0.5; eta=0.2; zeta=0.5; rho=0.01
    max_neigh = {
      ("legislator_term","voted_on","bill_version"):[128,64],
      ("bill_version","is_version","bill"):[2, 2],
      ("bill_version","priorVersion","bill_version"):[2, 2],
      ("bill_version","read","committee"):[128, 64],
      ("legislator_term","member_of","committee"):[128, 64],
      ("legislator_term","wrote","bill_version"):[128, 64],
      ("donor","donated_to","legislator_term"):[24, 8],
      ("lobby_firm","lobbied","legislator_term"):[24, 8],
      ("lobby_firm","lobbied","committee"):[24,8],
      ("topic","has","bill"):[128,64],
    }

# manual time col maps provided
MANUAL_NODE_TCOLS = {'bill':[-1],'bill_version':[-1],'legislator_term':[-1]}
MANUAL_EDGE_TCOLS = {
 ('legislator_term','voted_on','bill_version'):[-1],
 ('committee','rev_read','bill_version'):[-1],
 ('donor','donated_to','legislator_term'):[-1],
 ('lobby_firm','lobbied','committee'):[-1],
 ('lobby_firm','lobbied','legislator_term'):[-1],
 ('bill_version','read','committee'):[-1],
}

# ------------------------------- Utils ---------------------------------------
def nanfix(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _get_store(g, et):
    return g[et] if et in g.edge_types else None

def idx_on(x: torch.Tensor, like: torch.Tensor):
    x = x.to(like.device)
    if x.dtype != torch.long: x = x.long()
    return x

class Time2Vec(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.w0 = nn.Linear(1,1); self.wk = nn.Linear(1,k)
    def forward(self, t):
        t = t.view(-1,1)
        return torch.cat([self.w0(t), torch.sin(self.wk(t))], -1)

def mlp(dims):
    m=[]
    for i in range(len(dims)-1):
        m += [nn.Linear(dims[i], dims[i+1])]
        if i < len(dims)-2: m += [nn.ReLU()]
    return nn.Sequential(*m)

def balanced_ce(logits, target, class_weights=None, num_classes=3, label_smoothing=0.0):
    m = target.ge(0)
    if not m.any(): return logits.sum()*0
    logits, target = logits[m], target[m].long()
    one = F.one_hot(target, num_classes).float()
    if label_smoothing>0: one = (1-label_smoothing)*one + label_smoothing/num_classes
    logp = F.log_softmax(logits, -1)
    if class_weights is None: w = torch.ones(num_classes, device=logits.device)
    else: w = class_weights.to(logits.device)
    return (-(w[target] * (one*logp).sum(-1))).mean()

def brier(logits, target, C=3):
    m = target.ge(0)
    if not m.any(): return logits.sum()*0
    p = F.softmax(logits[m], -1)
    y = F.one_hot(target[m].long(), C).float()
    return ((p-y)**2).mean()

def ece(logits, target, bins=15):
    m = target.ge(0)
    if not m.any(): return torch.tensor(0., device=logits.device)
    p = F.softmax(logits[m], -1)
    conf, pred = p.max(-1)
    corr = pred.eq(target[m].long()).float()
    s = torch.tensor(0., device=logits.device)
    for i in range(bins):
        lo, hi = i/bins, (i+1)/bins
        mask = (conf>=lo)&(conf<hi)
        if mask.any():
            s += mask.float().mean() * (corr[mask].mean()-conf[mask].mean()).abs()
    return s

def macro_f1(logits, target):
    m = target.ge(0)
    if not m.any(): return torch.tensor(0., device=logits.device)
    logits, target = logits[m], target[m]
    pred = logits.argmax(-1)
    C = logits.size(-1); f1=0.
    for c in range(C):
        tp = ((pred==c)&(target==c)).sum().float()
        fp = ((pred==c)&(target!=c)).sum().float()
        fn = ((pred!=c)&(target==c)).sum().float()
        p = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6)
        f1 += 2*p*r/(p+r+1e-6)
    return f1/C

def orth_penalty(E):
    n = F.normalize(E, -1)
    G = n @ n.t()
    I = torch.eye(G.size(0), device=E.device)
    return ((G-I)**2).mean()

# --------------------------- Topic builder -----------------------------------
class TopicBuilder:
    def __init__(self, expected=None):
        self.expected = expected
    def __call__(self, data, bill_labels=None):
        if hasattr(data["bill"], "cluster"):
            raw = data["bill"].cluster.long()
        elif bill_labels is not None:
            raw = torch.full((data["bill"].num_nodes,), -1, dtype=torch.long)
            for bid, t in bill_labels.items():
                if 0<=bid<raw.numel(): raw[bid]=int(t)
        else:
            raise RuntimeError("No topic labels found")
        mask = raw.ge(0)
        uniq = torch.unique(raw[mask], sorted=True)
        remap = {int(t):i for i,t in enumerate(uniq.tolist())}
        topic_ix = torch.full_like(raw, -1)
        topic_ix[mask] = torch.tensor([remap[int(v)] for v in raw[mask].tolist()])
        T = len(uniq)
        if self.expected is not None and T != self.expected:
            print(f"[TopicBuilder] Warning: discovered {T} topics (expected {self.expected}). Proceeding with {T}.")
        data["topic"].num_nodes = T
        data["bill"].topic_ix = topic_ix
        src = topic_ix[mask]; dst = torch.nonzero(mask, as_tuple=False).view(-1)
        data[("topic","has","bill")].edge_index = torch.stack([src, dst], 0)
        data[("topic","has","bill")].edge_attr = torch.ones(dst.numel(),1)
        counts = torch.bincount(topic_ix[topic_ix>=0], minlength=T).float()
        prev = (counts / counts.sum().clamp_min(1)).to(torch.float32)
        data["topic"].prev = prev
        return T, topic_ix

# ------------------------ Feature projection ---------------------------------
class FeatureProjector(nn.Module):
    def __init__(self, data, d):
        super().__init__()
        self.d = d
        def dim(nt, default):
            x = getattr(data[nt], "x", None)
            return x.size(-1) if x is not None else default
        self.p = nn.ModuleDict({
            "bill": nn.Linear(dim("bill",770), d),
            "bill_version": nn.Linear(dim("bill_version",390), d),
            "legislator": nn.Linear(dim("legislator",385), d),
            "legislator_term": mlp([dim("legislator_term",4),64,d]),
            "committee": mlp([dim("committee",65),128,d]),
            "lobby_firm": nn.Linear(dim("lobby_firm",384), d),
            "donor": mlp([dim("donor",64),128,d]),
        })
    def forward(self, batch):
        out={}
        for nt in batch.node_types:
            if nt=="topic": continue
            x = getattr(batch[nt], "x", None)
            n = batch[nt].num_nodes
            x = torch.zeros(n, self.d, device=DEVICE) if x is None else x.to(DEVICE)
            x = nanfix(x)
            proj = self.p[nt] if nt in self.p else None
            out[nt] = proj(x) if proj is not None else x
        return out

# ------------------------ Backbone (Het-SAGE + time) -------------------------
class HetSAGE(nn.Module):
    def __init__(self, metadata, d, layers, drop, edge_dim):
        super().__init__()
        nts, ets = metadata
        self.nts = list(nts); self.ets = list(ets)
        self.convs = nn.ModuleList([nn.ModuleDict({ "__".join(et): nn.Linear(2*d, d) for et in self.ets }) for _ in range(layers)])
        self.edge_mlps = nn.ModuleDict({ "__".join(et): mlp([edge_dim, d]) for et in self.ets })
        self.norms = nn.ModuleDict({ nt: nn.LayerNorm(d) for nt in self.nts })
        self.drop = nn.Dropout(drop); self.layers=layers; self.d=d
    def forward(self, h, batch, edge_time):
        dev = next(iter(h.values())).device; d=self.d
        for l in range(self.layers):
            out = {nt: h[nt].new_zeros(h[nt].size(0), d) for nt in self.nts if nt in h}
            for et in self.ets:
                if et[0] not in h or et[2] not in h: continue
                s = _get_store(batch, et)
                if s is None or s.edge_index.numel()==0: continue
                ei = s.edge_index.to(dev).long()
                xs = h[et[0]]; xd = h[et[2]]
                if xs.size(0)==0 or xd.size(0)==0: continue
                agg = scatter_mean(xs.index_select(0, ei[0]), ei[1], dim=0, dim_size=xd.size(0))
                cat = torch.cat([xd, agg], -1)
                y = self.convs[l]["__".join(et)](cat)
                ft = edge_time.get(et, None)
                if ft is not None and ft.size(0)==ei.size(1):
                    bias = self.edge_mlps["__".join(et)](ft.to(dev))
                    add = scatter_mean(bias, ei[1], dim=0, dim_size=xd.size(0))
                    y = y + add
                out[et[2]] = out[et[2]] + y
            h = {nt: self.norms[nt](self.drop(out[nt])) for nt in out}
        return h

# ------------------------ Time helpers ---------------------------------------
def _select_time_col(store, et):
    ea = getattr(store, "edge_attr", None)
    if ea is None: return None
    if et in MANUAL_EDGE_TCOLS:
        cols = MANUAL_EDGE_TCOLS[et]
        v = ea[:, cols] if ea.dim()==2 else ea[cols]
        v = v.float()
        if v.dim()==2: v = v.mean(-1)
        return v
    if ea.dim()==2 and ea.size(1)>0: return ea[:,-1].float()
    return None

def build_edge_time_feats(batch, t2v):
    feats={}
    for et in batch.edge_types:
        s = _get_store(batch, et)
        if s is None or s.edge_index.numel()==0: feats[et]=None; continue
        dt = _select_time_col(s, et)
        if dt is None: feats[et]=None; continue
        feats[et] = t2v(dt.to(DEVICE))
    return feats

# ------------------------ Context fuse ---------------------------------------
def masked_pool(src_feat, src_idx, dst_idx, dst_size, edge_time=None, cutoff=None):
    if edge_time is not None and cutoff is not None:
        m = edge_time.le(cutoff)
        if m.sum().item()==0:
            return torch.zeros(dst_size, src_feat.size(-1), device=src_feat.device)
        src_feat = src_feat[m]; src_idx = src_idx[m]; dst_idx = dst_idx[m]
    if src_idx.numel()==0: return torch.zeros(dst_size, src_feat.size(-1), device=src_feat.device)
    return scatter_mean(src_feat, dst_idx, dim=0, dim_size=dst_size)

class GatedFuse(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Linear(2*d, d); self.out = nn.Linear(d, d)
    def forward(self, a, b):
        g = torch.sigmoid(self.g(torch.cat([a,b],-1)))
        return self.out(g*a + (1-g)*b)

class MetaPathAgg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fuse = mlp([6*d, d])
    def forward(self, batch, h, vote_lt, vote_bv, topic_ix, t2v_edgefeats, vote_time):
        dev = next(iter(h.values())).device; d=h["bill"].size(-1)
        E = vote_lt.numel()
        if E==0: return torch.zeros(0,d,device=dev)
        # bv->bill
        eb = _get_store(batch, ("bill_version","is_version","bill"))
        if eb is None or eb.edge_index.numel()==0:
            b_of_bv = batch["bill_version"].n_id[vote_bv]  # local->global ids
            b_of_bv = batch._global_bv2bill.index_select(0, b_of_bv.to(batch._global_bv2bill.device))
            b_local_map = {int(g):i for i,g in enumerate(batch["bill"].n_id.tolist())}
            b_of_bv = torch.tensor([b_local_map.get(int(x), 0) for x in b_of_bv.tolist()], device=dev)
        else:
            b_of_bv = eb.edge_index[1][vote_bv]
        b_of_bv = idx_on(b_of_bv, h["bill"])
        # LT context
        lt_ctx = h["legislator_term"].index_select(0, idx_on(vote_lt, h["legislator_term"]))
        # prior versions -> current bv
        pv = _get_store(batch, ("bill_version","priorVersion","bill_version"))
        if pv is not None and pv.edge_index.numel()>0:
            pv_src, pv_dst = pv.edge_index
            Etime = t2v_edgefeats.get(("bill_version","priorVersion","bill_version"), None)
            pool = masked_pool(h["bill_version"].index_select(0, idx_on(pv_src, h["bill_version"])),
                               idx_on(pv_src, h["bill_version"]), idx_on(pv_dst, h["bill_version"]),
                               h["bill_version"].size(0), Etime, None)
            prior_ctx = pool.index_select(0, idx_on(vote_bv, h["bill_version"]))
        else:
            prior_ctx = torch.zeros(E,d,device=dev)
        # reads committee -> bv -> bill
        rd = _get_store(batch, ("bill_version","read","committee"))
        if rd is not None and rd.edge_index.numel()>0 and h["committee"].size(0)>0:
            r_src, r_dst = rd.edge_index
            r_time = t2v_edgefeats.get(("bill_version","read","committee"), None)
            comm_on_edges = h["committee"].index_select(0, idx_on(r_dst, h["committee"]))
            bv_pool = masked_pool(comm_on_edges, idx_on(r_dst, h["committee"]), idx_on(r_src, h["bill_version"]),
                                  h["bill_version"].size(0), r_time, None)
            bv2b = _get_store(batch, ("bill_version","is_version","bill"))
            if bv2b is None or bv2b.edge_index.numel()==0:
                bill_comm = torch.zeros(h["bill"].size(0), d, device=dev)
            else:
                bill_comm = scatter_mean(bv_pool.index_select(0, idx_on(bv2b.edge_index[0], h["bill_version"])),
                                         idx_on(bv2b.edge_index[1], h["bill"]), dim=0, dim_size=h["bill"].size(0))
            committee_ctx = bill_comm.index_select(0, b_of_bv)
        else:
            committee_ctx = torch.zeros(E,d,device=dev)
        # LT committee profile
        mem = _get_store(batch, ("legislator_term","member_of","committee"))
        if mem is not None and mem.edge_index.numel()>0 and h["committee"].size(0)>0:
            m_src, m_dst = mem.edge_index
            m_time = t2v_edgefeats.get(("legislator_term","member_of","committee"), None)
            feat = h["committee"].index_select(0, idx_on(m_dst, h["committee"]))
            lt_comm = masked_pool(feat, idx_on(m_dst, h["committee"]), idx_on(m_src, h["legislator_term"]),
                                  h["legislator_term"].size(0), m_time, None)
            lt_committee = lt_comm.index_select(0, idx_on(vote_lt, h["legislator_term"]))
        else:
            lt_committee = torch.zeros(E,d,device=dev)
        # donor/lobby
        donor_rel = _get_store(batch, ("donor","donated_to","legislator_term"))
        if donor_rel is not None and donor_rel.edge_index.numel()>0 and h["donor"].size(0)>0:
            ds, dd = donor_rel.edge_index
            d_time = t2v_edgefeats.get(("donor","donated_to","legislator_term"), None)
            feat = h["donor"].index_select(0, idx_on(ds, h["donor"]))
            lt_don = masked_pool(feat, idx_on(ds, h["donor"]), idx_on(dd, h["legislator_term"]),
                                 h["legislator_term"].size(0), d_time, None).index_select(0, idx_on(vote_lt, h["legislator_term"]))
        else:
            lt_don = torch.zeros(E,d,device=dev)
        lob_lt = _get_store(batch, ("lobby_firm","lobbied","legislator_term"))
        if lob_lt is not None and lob_lt.edge_index.numel()>0 and h["lobby_firm"].size(0)>0:
            ls, ld = lob_lt.edge_index
            l_time = t2v_edgefeats.get(("lobby_firm","lobbied","legislator_term"), None)
            feat = h["lobby_firm"].index_select(0, idx_on(ls, h["lobby_firm"]))
            lt_lob = masked_pool(feat, idx_on(ls, h["lobby_firm"]), idx_on(ld, h["legislator_term"]),
                                 h["legislator_term"].size(0), l_time, None).index_select(0, idx_on(vote_lt, h["legislator_term"]))
        else:
            lt_lob = torch.zeros(E,d,device=dev)
        # topic context
        topic_ix_dev = topic_ix.to(h["topic"].device)
        topic_idx = topic_ix_dev[b_of_bv].clamp(min=0)
        topic_ctx = h["topic"].index_select(0, topic_idx)
        return self.fuse(torch.cat([lt_ctx, prior_ctx, committee_ctx, lt_committee, lt_don+lt_lob, topic_ctx], -1))

# ------------------------------ Heads ----------------------------------------
class VoteHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = mlp([3*d, 2*d, d])
        self.out = nn.Linear(d, 3)
        self.gate = GatedFuse(d)
    def forward(self, lt, bill, topic, ctx):
        z = self.gate(lt, bill)
        x = self.net(torch.cat([z, topic, ctx], -1))
        return self.out(x)

class OutcomeHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = mlp([2*d+1, d]); self.out = nn.Linear(d,3)
    def forward(self, bill, comm_topic_ctx, margin1):
        x = torch.cat([bill, comm_topic_ctx, margin1], -1)
        return self.out(self.net(x))

class GateHead(nn.Module):
    def __init__(self, d, stages=6):
        super().__init__()
        self.net = mlp([3*d, d]); self.out = nn.Linear(d, stages)
    def forward(self, committee, bill, topic):
        return self.out(self.net(torch.cat([committee, bill, topic], -1)))

class StanceHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = mlp([2*d, d]); self.out = nn.Linear(d,1)
    def forward(self, actor, topic):
        return torch.tanh(self.out(self.net(torch.cat([actor, topic], -1))))

class MaskNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = mlp([2*d, d, 1])
    def forward(self, actor, topic, temp=0.5, training=True):
        logits = self.net(torch.cat([actor, topic], -1)).squeeze(-1)
        if training:
            u = torch.rand_like(logits).clamp_min(1e-6)
            g = -torch.log(-torch.log(u))
            logits = (logits + g)/temp
        return torch.sigmoid(logits)

# ------------------------------- Model ---------------------------------------
class LeGNN4(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self.cfg = cfg
        self.proj = FeatureProjector(data, cfg.d)
        self.topic_emb = nn.Embedding(data["topic"].num_nodes, cfg.d)
        self.t2v = Time2Vec(cfg.time2vec_k)
        self.backbone = HetSAGE(data.metadata(), cfg.d, cfg.layers, cfg.drop, cfg.time2vec_k+1)
        self.metapath = MetaPathAgg(cfg.d)
        self.vote_head = VoteHead(cfg.d)
        self.out_head = OutcomeHead(cfg.d)
        self.gate_head = GateHead(cfg.d)
        self.stance_lt = StanceHead(cfg.d)
        self.stance_donor = StanceHead(cfg.d)
        self.stance_lobby = StanceHead(cfg.d)
        self.mask_net = MaskNet(cfg.d)
    def encode(self, batch):
        raw = self.proj(batch)
        h = {nt: nanfix(raw.get(nt, torch.zeros(batch[nt].num_nodes, self.cfg.d,device=DEVICE))) for nt in batch.node_types if nt!="topic"}
        h["topic"] = self.topic_emb.weight
        etime = build_edge_time_feats(batch, self.t2v)
        h = self.backbone(h, batch, etime)
        return {k: torch.clamp(v, -50, 50) for k,v in h.items()}
    def vote_forward(self, batch, h, topic_ix, class_time=None):
        e = _get_store(batch, ("legislator_term","voted_on","bill_version"))
        if e is None or e.edge_index.numel()==0: return torch.zeros(0,3,device=DEVICE)
        lt_i, bv_i = e.edge_index
        eb = _get_store(batch, ("bill_version","is_version","bill"))
        if eb is not None and eb.edge_index.numel()>0:
            bill_i = eb.edge_index[1][bv_i]
        else:
            bill_i = batch._global_bv2bill.index_select(0, batch["bill_version"].n_id[bv_i].to(batch._global_bv2bill.device))
        bill_i = idx_on(bill_i, h["bill"])
        topic_i = topic_ix.to(h["topic"].device)[bill_i].clamp(min=0)
        t2v = build_edge_time_feats(batch, self.t2v)
        vt = None
        if hasattr(e, "edge_attr") and e.edge_attr is not None and e.edge_attr.size(-1)>0:
            col = MANUAL_EDGE_TCOLS.get(("legislator_term","voted_on","bill_version"), [-1])[0]
            vt = e.edge_attr[:, col].float().to(DEVICE)
        ctx = self.metapath(batch, h, lt_i, bv_i, topic_ix, t2v, vt)
        return self.vote_head(h["legislator_term"].index_select(0, idx_on(lt_i, h["legislator_term"])),
                              h["bill"].index_select(0, bill_i),
                              h["topic"].index_select(0, topic_i),
                              ctx)
    def outcome_forward(self, batch, h, vote_margin1, topic_ix):
        rd = _get_store(batch, ("bill_version","read","committee"))
        if rd is not None and rd.edge_index.numel()>0:
            r_src, r_dst = rd.edge_index
            bv_pool = scatter_mean(h["committee"].index_select(0, idx_on(r_dst, h["committee"])),
                                   idx_on(r_src, h["bill_version"]), dim=0, dim_size=h["bill_version"].size(0))
            eb = _get_store(batch, ("bill_version","is_version","bill"))
            if eb is not None and eb.edge_index.numel()>0:
                bill_comm = scatter_mean(bv_pool.index_select(0, idx_on(eb.edge_index[0], h["bill_version"])),
                                         idx_on(eb.edge_index[1], h["bill"]), dim=0, dim_size=h["bill"].size(0))
            else:
                bill_comm = torch.zeros_like(h["bill"])
        else:
            bill_comm = torch.zeros_like(h["bill"])
        bill_topic = h["topic"].index_select(0, topic_ix.to(h["topic"].device).clamp(min=0))
        return self.out_head(h["bill"], bill_comm + bill_topic, vote_margin1)

# --------------------------- Label builders ----------------------------------
def build_vote_class_weights(data):
    e = data[("legislator_term","voted_on","bill_version")]
    y = e.edge_attr[:,0].long()
    vals = torch.tensor([-1,0,1], dtype=torch.long)
    cnt = torch.stack([(y==v).sum() for v in vals]).float().clamp_min(1)
    w = cnt.sum()/cnt; w = w/w.mean()
    return torch.tensor([w[0], w[1], w[2]])

def build_stance_labels(data, topic_ix, min_eff=5, lam=1/365.0):
    e = data[("legislator_term","voted_on","bill_version")]
    labels = e.edge_attr[:,0].float()
    bv2b = data[("bill_version","is_version","bill")].edge_index[1]
    bill = bv2b[e.edge_index[1]]
    topic = topic_ix[bill]
    m = labels.ne(0) & topic.ge(0)
    lt = e.edge_index[0][m]; t = topic[m]; y = labels[m]
    if e.edge_attr.size(-1)>0:
        col = MANUAL_EDGE_TCOLS.get(("legislator_term","voted_on","bill_version"), [-1])[0]
        age = e.edge_attr[m, col].float()
        w = torch.exp(-lam * age)
    else:
        w = torch.ones_like(y)
    Nlt = data["legislator_term"].num_nodes; T = int(topic_ix.max().item()+1)
    lin = lt*T + t
    num = scatter_add(w*y, lin, dim=0, dim_size=Nlt*T).view(Nlt, T)
    den = scatter_add(w, lin, dim=0, dim_size=Nlt*T).view(Nlt, T)
    eff = scatter_add(torch.ones_like(w), lin, dim=0, dim_size=Nlt*T).view(Nlt, T)
    stance = num/den.clamp_min(1e-6)
    stance[eff < min_eff] = float("nan")
    weights = den / den.max().clamp_min(1)
    return stance, weights

# ------------------------- Neighbor helpers ----------------------------------
def build_neighbors_dict(g, base):
    exist = set(g.edge_types)
    out={}
    for k,v in base.items():
        if k in exist: out[k]=v
    return out

def _rev_of(et):
    return (et[2], f"rev_{et[1]}", et[0])

def make_num_neighbors_covering_all(g, overrides, hops=3, default=0, mirror_revs=True):
    out = {('*','*','*'): [default]*hops}
    for et in g.edge_types:
        out[et] = [default]*hops
    for k, v in list(overrides.items()):
        if k in g.edge_types:
            out[k] = v
        if mirror_revs:
            rk = _rev_of(k)
            if rk in g.edge_types and rk not in overrides:
                out[rk] = v
    return out

def map_bv_to_bill(batch, bv_idx_cpu, global_bv2bill_cpu):
    bv_global = batch["bill_version"].n_id[bv_idx_cpu].cpu()
    bill_global = global_bv2bill_cpu.index_select(0, bv_global)
    if "bill" in batch.node_types:
        b_nid = batch["bill"].n_id.cpu().tolist()
        g2l = {int(g): i for i, g in enumerate(b_nid)}
        bill_local_list = [g2l.get(int(g), -1) for g in bill_global.tolist()]
        bill_local = torch.tensor(bill_local_list, dtype=torch.long)
    else:
        bill_local = torch.full((bill_global.numel(),), -1, dtype=torch.long)
    return bill_local, bill_global

# ------------------------------- Trainer -------------------------------------
class Trainer:
    def __init__(self, data, cfg):
        self.cfg = cfg
        base = data.clone()
        base = ToUndirected()(base)
        base = RemoveIsolatedNodes()(base)
        for et in base.edge_types:
            base[et].edge_index = base[et].edge_index.long()
        eb = base[("bill_version","is_version","bill")].edge_index
        self.global_bv2bill = torch.zeros(base["bill_version"].num_nodes, dtype=torch.long)
        self.global_bv2bill[eb[0]] = eb[1]
        self.topic_builder = TopicBuilder(expected=cfg.topics_expected)
        self.T, self.topic_ix = self.topic_builder(base)
        base["topic"].x = torch.eye(self.T)
        self.data = base

        vote_overrides = {
            k: cfg.max_neigh[k] for k in
            [("legislator_term","voted_on","bill_version"), ('bill_version', 'is_version', 'bill'), ("bill_version","priorVersion","bill_version"), ("bill_version","read","committee"), ("legislator_term","member_of","committee"), ("legislator_term","wrote","bill_version"), ("donor","donated_to","legislator_term"), ("lobby_firm","lobbied","legislator_term"), ("lobby_firm","lobbied","committee"), ("topic","has","bill")]
        }
        vote_nei = make_num_neighbors_covering_all(self.data, vote_overrides, hops=2, default=0, mirror_revs=True)

        bill_overrides = {
            k: cfg.max_neigh[k] for k in [
            ("bill_version","is_version","bill"),
            ("bill_version","read","committee"),
            ("legislator_term","voted_on","bill_version"),
            ("topic","has","bill"),
            ("legislator_term","wrote","bill_version")]
        }
        bill_nei = make_num_neighbors_covering_all(self.data, bill_overrides, hops=2, default=0, mirror_revs=True)

        gate_overrides = {
            k: cfg.max_neigh[k] for k in [
            ("bill_version","read","committee"),
            ("bill_version","is_version","bill"),
            ("topic","has","bill"),
            ("legislator_term","member_of","committee")]
        }
        gate_nei = make_num_neighbors_covering_all(self.data, gate_overrides, hops=2, default=0, mirror_revs=True)

        self.vote_loader = NeighborLoader(self.data,
            input_nodes=("legislator_term", torch.arange(self.data["legislator_term"].num_nodes)),
            num_neighbors=vote_nei, batch_size=cfg.vote_bsz, shuffle=True,
            num_workers=max(1,(os.cpu_count() or 2)//2), pin_memory=(DEVICE.type=="cuda"))
        self.bill_loader = NeighborLoader(self.data,
            input_nodes=("bill", torch.arange(self.data["bill"].num_nodes)),
            num_neighbors=bill_nei, batch_size=cfg.bill_bsz, shuffle=True,
            num_workers=max(1,(os.cpu_count() or 2)//2), pin_memory=(DEVICE.type=="cuda"))
        self.gate_loader = NeighborLoader(self.data,
            input_nodes=("committee", torch.arange(self.data["committee"].num_nodes)),
            num_neighbors=gate_nei, batch_size=cfg.gate_bsz, shuffle=True,
            num_workers=max(1,(os.cpu_count() or 2)//2), pin_memory=(DEVICE.type=="cuda"))

        self.model = LeGNN4(self.data, cfg).to(DEVICE)
        self.model.topic_emb = nn.Embedding(self.T, cfg.d).to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.vote_class_w = build_vote_class_weights(self.data)
        self.stance_lbl, self.stance_w = build_stance_labels(self.data, self.topic_ix)
    def _encode_batch(self, batch):
        batch = batch.to(DEVICE, non_blocking=True)
        for et in batch.edge_types: batch[et].edge_index = batch[et].edge_index.long()
        batch._global_bv2bill = self.global_bv2bill.to(DEVICE)
        return self.model.encode(batch), batch
    def _vote_step(self, batch):
        h, batch = self._encode_batch(batch)
        local_topic_ix = self.topic_ix[batch["bill"].n_id.cpu()].to(DEVICE)
        logits = self.model.vote_forward(batch, h, local_topic_ix)
        e = batch[("legislator_term","voted_on","bill_version")]
        target = e.edge_attr[:,0].to(DEVICE).long()
        loss_v = balanced_ce(logits, target, self.vote_class_w.to(DEVICE), label_smoothing=self.cfg.ls) + 0.1*brier(logits, target)
        p = F.softmax(logits, -1)
        lt_i, bv_i = e.edge_index
        eb = _get_store(batch, ("bill_version","is_version","bill"))
        bill_i = (eb.edge_index[1][bv_i] if eb is not None and eb.edge_index.numel()>0
                  else self.global_bv2bill.index_select(0, batch["bill_version"].n_id[bv_i].cpu()).to(DEVICE))
        margin_part = scatter_add((p[:,2]-p[:,0]), bill_i, dim=0, dim_size=batch["bill"].num_nodes)
        topic_i = local_topic_ix[bill_i].clamp(min=0)
        masks = self.model.mask_net(h["legislator_term"].index_select(0, lt_i), h["topic"].index_select(0, topic_i), training=True)
        loss_mask = (masks * (p[:,2]-p[:,0]).abs()).mean()
        topic_vec = h["topic"].index_select(0, topic_i)
        con = 0.5*(1 - F.cosine_similarity(h["bill"].index_select(0, bill_i), topic_vec)).mean()
        loss = self.cfg.alpha*loss_v + self.cfg.eta*con + self.cfg.zeta*loss_mask + self.cfg.rho*orth_penalty(self.model.topic_emb.weight)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            return {
                "loss_vote": float(loss_v.item()),
                "loss_mask": float(loss_mask.item()),
                "loss_con": float(con.item()),
                "ece": float(ece(logits, target, bins=self.cfg.ece_bins).item()),
                "bill_margin_part": (batch["bill"].n_id.cpu(), margin_part.detach().cpu()),
            }
    def _outcome_step(self, batch, bill_margin_cache):
        h, batch = self._encode_batch(batch)
        bill_nid_cpu = batch["bill"].n_id.cpu()
        margin1 = bill_margin_cache.index_select(0, bill_nid_cpu).to(DEVICE, non_blocking=False).view(-1, 1)
        logits = self.model.outcome_forward(batch, h, margin1, self.topic_ix[bill_nid_cpu].to(DEVICE))
        target = batch["bill"].y.to(DEVICE).long()
        loss = balanced_ce(logits, target, label_smoothing=self.cfg.ls) + 0.1*brier(logits, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            return {
                "loss_out": float(loss.item()),
                "out_f1": float(macro_f1(logits, target).item())
            }
    def _gate_step(self, batch):
        h, batch = self._encode_batch(batch)
        rd = _get_store(batch, ("bill_version","read","committee"))
        if rd is None or rd.edge_index.numel() == 0:
            return {"loss_gate": 0.0}
        bv_i, c_i = rd.edge_index
        bill_local, bill_global = map_bv_to_bill(batch, bv_i.cpu(), self.global_bv2bill)
        m = bill_local.ge(0)
        if not m.any(): return {"loss_gate": 0.0}
        bv_i = bv_i[m]; c_i = c_i[m]; bill_local = bill_local[m]; bill_global = bill_global[m]
        t_i = self.topic_ix.index_select(0, bill_global).clamp(min=0).to(DEVICE)
        comm_h  = h["committee"].index_select(0, c_i)
        bill_h  = h["bill"].index_select(0, bill_local.to(DEVICE))
        topic_h = h["topic"].index_select(0, t_i)
        logits = self.model.gate_head(comm_h, bill_h, topic_h)
        if getattr(rd, "edge_attr", None) is not None and rd.edge_attr.size(-1) > 0:
            tgt_all = rd.edge_attr[:, 0].long()
            tgt = tgt_all[m].to(DEVICE).clamp(0, logits.size(-1)-1)
        else:
            tgt = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
        loss = balanced_ce(logits, tgt, num_classes=logits.size(-1), label_smoothing=self.cfg.ls)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"loss_gate": float(loss.item())}
    def _stance_step(self):
        Nlt = self.data["legislator_term"].num_nodes; T = self.T
        lt_idx = torch.randint(0, Nlt, (self.cfg.eval_bsz,))
        t_idx = torch.randint(0, T, (self.cfg.eval_bsz,))
        overrides = {
            k: [self.cfg.max_neigh[k][0], 0] for k in [
            ("legislator_term","member_of","committee"),
            ("legislator_term","voted_on","bill_version"),
            ("legislator_term","wrote","bill_version"),
            ("donor","donated_to","legislator_term"),
            ("lobby_firm","lobbied","legislator_term"),
            ("topic","has","bill")]}
        overrides.update({
            ("legislator", "samePerson", "legislator_term"): [3, 0]
        })
        nei = make_num_neighbors_covering_all(self.data, overrides, hops=2, default=0, mirror_revs=True)
        loader = NeighborLoader(self.data,
            input_nodes=("legislator_term", lt_idx),
            num_neighbors=nei, batch_size=self.cfg.eval_bsz, shuffle=False,
            num_workers=0)
        batch = next(iter(loader))
        h = self.model.encode(batch)
        nid_map = {int(g):i for i,g in enumerate(batch["legislator_term"].n_id.tolist())}
        lt_loc = torch.tensor([nid_map.get(int(i), 0) for i in lt_idx.tolist()], device=DEVICE)
        t_idx_clamped = t_idx.clamp_max(self.T - 1)
        lbl_rows = self.stance_lbl.index_select(0, lt_idx)
        w_rows = self.stance_w.index_select(0, lt_idx)
        s_lbl = lbl_rows.gather(1, t_idx_clamped.view(-1, 1)).squeeze(1)
        s_w = w_rows.gather(1, t_idx_clamped.view(-1, 1)).squeeze(1)
        actor_h = h["legislator_term"].index_select(0, lt_loc)
        topic_h = self.model.topic_emb.weight.index_select(0, t_idx_clamped.to(DEVICE))
        pred = self.model.stance_lt(actor_h, topic_h).squeeze(-1)
        m = torch.isfinite(s_lbl)
        if not m.any(): return {"loss_stance": 0.0}
        pred_m = pred[m]; target_m = s_lbl[m].to(DEVICE); w_m = s_w[m].to(DEVICE)
        w_m = w_m / w_m.mean().clamp_min(1e-6)
        loss = ((pred_m - target_m) ** 2 * w_m).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"loss_stance": float(loss.item())}
    def train_epoch(self):
        stats = {}; bill_margin_cache = torch.zeros(self.data["bill"].num_nodes, dtype=torch.float32)
        for batch in tqdm(self.vote_loader, desc="Vote Batches", leave=False):
            s = self._vote_step(batch)
            for k,v in s.items():
                if k=="bill_margin_part": continue
                stats[k]=stats.get(k,0.0)+float(v)
            nid, mpart = s["bill_margin_part"]; bill_margin_cache.index_add_(0, nid, mpart)
        for batch in tqdm(self.bill_loader, desc="Bill Batches", leave=False):
            s = self._outcome_step(batch, bill_margin_cache)
            for k,v in s.items(): stats[k]=stats.get(k,0.0)+float(v)
        for batch in tqdm(self.gate_loader, desc="Gate Batches", leave=False):
            s = self._gate_step(batch)
            for k,v in s.items(): stats[k]=stats.get(k,0.0)+float(v)
        s = self._stance_step()
        for k,v in s.items(): stats[k]=stats.get(k,0.0)+float(v)
        return stats
    @torch.no_grad()
    def evaluate(self, steps=5):
        self.model.eval()
        ve=0.; n=0
        for batch in self.vote_loader:
            h, batch = self._encode_batch(batch)
            logits = self.model.vote_forward(batch, h, self.topic_ix[batch["bill"].n_id.cpu()].to(DEVICE))
            e = batch[("legislator_term","voted_on","bill_version")]
            tgt = e.edge_attr[:,0].to(DEVICE).long()
            ve += float(ece(logits, tgt, bins=self.cfg.ece_bins).item())
            n += 1
            if n>=steps: break
        if n>0:
            ve/=n
        self.model.train()
        return {"eval_vote_ece": ve}
    @torch.no_grad()
    def embed_full(self):
        self.model.eval()
        out={}
        for nt in self.data.node_types:
            N = self.data[nt].num_nodes
            store = torch.zeros(N, self.cfg.d)
            loader = NeighborLoader(self.data, input_nodes=(nt, torch.arange(N)), num_neighbors=build_neighbors_dict(self.data, {
                ("bill_version","is_version","bill"):[24,0],
                ("legislator_term","voted_on","bill_version"):[24,0],
                ("bill_version","read","committee"):[24,0],
                ("topic","has","bill"):[24,0],
            }), batch_size=2048, shuffle=False, num_workers=0)
            for b in loader:
                h = self.model.encode(b)
                if nt not in h: continue
                store[b[nt].n_id] = h[nt].cpu()
            out[nt]=store
        return out

# ------------------------------ Outputs --------------------------------------
@torch.no_grad()
def _safe_topic_ix_for_bills(topic_ix_cpu, bill_global_cpu):
    m = bill_global_cpu.ge(0) & bill_global_cpu.lt(topic_ix_cpu.size(0))
    out = torch.full_like(bill_global_cpu, 0)
    out[m] = topic_ix_cpu.index_select(0, bill_global_cpu[m])
    return out

@torch.no_grad()
def _vote_edge_pass(model, data, cfg, global_bv2bill_cpu, topic_ix_cpu):
    model.eval()
    loader = NeighborLoader(
        data,
        input_nodes=("legislator_term", torch.arange(data["legislator_term"].num_nodes)),
        num_neighbors=make_num_neighbors_covering_all(
            data,
            {
                ("legislator_term","voted_on","bill_version"):[32,24,16],
                ("bill_version","is_version","bill"):[10,8,4],
                ("bill_version","priorVersion","bill_version"):[4,4,4],
                ("bill_version","read","committee"):[6,4,0],
                ("legislator_term","member_of","committee"):[6,4,0],
                ("topic","has","bill"):[8,6,0],
                ("legislator_term","wrote","bill_version"):[6,4,0],
                ("donor","donated_to","legislator_term"):[8,4,0],
                ("lobby_firm","lobbied","legislator_term"):[8,4,0],
            },
            hops=3, default=0, mirror_revs=True
        ),
        batch_size=max(256, cfg.vote_bsz//2),
        shuffle=False, num_workers=0,
    )

    N_bill = data["bill"].num_nodes
    bill_margin = torch.zeros(N_bill, dtype=torch.float32)
    bill_yes = torch.zeros(N_bill, dtype=torch.float32)
    bill_no  = torch.zeros(N_bill, dtype=torch.float32)

    edges_lt = []; edges_bill = []; edges_topic = []; edges_yesprob = []; edges_noprob = []; edges_maskval = []

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=False)
        h = model.encode(batch)
        e = _get_store(batch, ("legislator_term","voted_on","bill_version"))
        if e is None or e.edge_index.numel()==0: continue
        lt_i, bv_i = e.edge_index
        bv_glob = batch["bill_version"].n_id[bv_i].cpu()
        bill_glob = global_bv2bill_cpu.index_select(0, bv_glob)

        if "bill" in batch.node_types:
            b_nid = batch["bill"].n_id.cpu().tolist()
            g2l = {int(g): i for i,g in enumerate(b_nid)}
            bill_loc = torch.tensor([g2l.get(int(g), -1) for g in bill_glob.tolist()], dtype=torch.long)
        else:
            bill_loc = torch.full((bill_glob.numel(),), -1, dtype=torch.long)
        m = bill_loc.ge(0)
        if not m.any(): continue

        lt_i   = lt_i[m]
        bv_i   = bv_i[m]
        bill_loc = bill_loc[m].to(DEVICE)
        bill_glob = bill_glob[m]
        topic_i = _safe_topic_ix_for_bills(topic_ix_cpu, bill_glob).clamp(min=0).to(DEVICE)

        # inputs for vote head
        lt_h   = h["legislator_term"].index_select(0, lt_i)
        bill_h = h["bill"].index_select(0, bill_loc)
        topic_h= h["topic"].index_select(0, topic_i)

        # metapath ctx (ensure topic_ix on device inside)
        t2v = build_edge_time_feats(batch, model.t2v)
        ctx = model.metapath(batch, h, lt_i, bv_i, topic_ix_cpu, t2v, None)

        logits = model.vote_head(lt_h, bill_h, topic_h, ctx)
        probs = F.softmax(logits, -1)
        yes, no = probs[:,2], probs[:,0]
        mvals = model.mask_net(lt_h, topic_h, training=False)

        part_yes = scatter_add(yes, bill_loc, dim=0, dim_size=h["bill"].size(0))
        part_no  = scatter_add(no , bill_loc, dim=0, dim_size=h["bill"].size(0))
        b_local_to_global = batch["bill"].n_id.cpu()
        bill_margin.index_add_(0, b_local_to_global, (part_yes - part_no).detach().cpu())
        bill_yes.index_add_(0, b_local_to_global, part_yes.detach().cpu())
        bill_no.index_add_(0,  b_local_to_global, part_no.detach().cpu())

        edges_lt.append(batch["legislator_term"].n_id[lt_i].cpu())
        edges_bill.append(bill_glob.cpu())
        edges_topic.append(topic_i.detach().cpu())
        edges_yesprob.append(yes.detach().cpu())
        edges_noprob.append(no.detach().cpu())
        edges_maskval.append(mvals.detach().cpu())

    if len(edges_lt)==0:
        return {
            "bill_margin": bill_margin,
            "edge_rows": {
                "lt": torch.zeros(0, dtype=torch.long),
                "bill": torch.zeros(0, dtype=torch.long),
                "topic": torch.zeros(0, dtype=torch.long),
                "p_yes": torch.zeros(0),
                "p_no": torch.zeros(0),
                "mask": torch.zeros(0),
            }
        }

    edge_rows = {
        "lt":   torch.cat(edges_lt,   0),
        "bill": torch.cat(edges_bill, 0),
        "topic":torch.cat(edges_topic,0),
        "p_yes":torch.cat(edges_yesprob,0),
        "p_no": torch.cat(edges_noprob, 0),
        "mask": torch.cat(edges_maskval,0),
    }
    return {"bill_margin": bill_margin, "edge_rows": edge_rows}

@torch.no_grad()
def _gate_edge_pass(model, data, cfg, global_bv2bill_cpu, topic_ix_cpu):
    model.eval()
    loader = NeighborLoader(
        data,
        input_nodes=("committee", torch.arange(data["committee"].num_nodes)),
        num_neighbors=make_num_neighbors_covering_all(
            data,
            {
                ("bill_version","read","committee"):[8,8,8],
                ("bill_version","is_version","bill"):[8,8,8],
                ("topic","has","bill"):[8,6,0],
                ("legislator_term","member_of","committee"):[8,6,0],
            },
            hops=3, default=0, mirror_revs=True
        ),
        batch_size=max(256, cfg.gate_bsz//2),
        shuffle=False, num_workers=0,
    )

    N_c = data["committee"].num_nodes
    comm_adv_sum = torch.zeros(N_c)
    comm_adv_cnt = torch.zeros(N_c)
    bill_bottlenecks = {}

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=False)
        h = model.encode(batch)
        rd = _get_store(batch, ("bill_version","read","committee"))
        if rd is None or rd.edge_index.numel()==0: continue
        bv_i, c_i = rd.edge_index
        bv_glob = batch["bill_version"].n_id[bv_i].cpu()
        bill_glob = global_bv2bill_cpu.index_select(0, bv_glob)
        if "bill" in batch.node_types:
            b_nid = batch["bill"].n_id.cpu().tolist()
            g2l = {int(g): i for i,g in enumerate(b_nid)}
            bill_loc = torch.tensor([g2l.get(int(g), -1) for g in bill_glob.tolist()], dtype=torch.long)
        else:
            bill_loc = torch.full((bill_glob.numel(),), -1, dtype=torch.long)
        m = bill_loc.ge(0)
        if not m.any(): continue
        bv_i = bv_i[m]; c_i = c_i[m]; bill_loc = bill_loc[m].to(DEVICE); bill_glob = bill_glob[m]
        t_i = _safe_topic_ix_for_bills(topic_ix_cpu, bill_glob).clamp(min=0).to(DEVICE)

        logits = model.gate_head(
            h["committee"].index_select(0, c_i),
            h["bill"].index_select(0, bill_loc),
            h["topic"].index_select(0, t_i),
        )
        p = F.softmax(logits, -1)
        p_adv = p.max(-1).values

        comm_global = batch["committee"].n_id[c_i].cpu()
        comm_adv_sum.index_add_(0, comm_global, p_adv.detach().cpu())
        comm_adv_cnt.index_add_(0, comm_global, torch.ones_like(p_adv.detach().cpu()))

        bill_ids = bill_glob.cpu().tolist()
        vals = p_adv.detach().cpu().tolist()
        comm_ids = comm_global.tolist()
        for b, c, v in zip(bill_ids, comm_ids, vals):
            bill_bottlenecks.setdefault(b, []).append({"committee_id": int(c), "p_gate": float(v)})

    comm_adv_cnt = comm_adv_cnt.clamp_min(1)
    gate_index = (comm_adv_sum / comm_adv_cnt)
    return {"gate_index": gate_index, "bill_bottlenecks": bill_bottlenecks}

@torch.no_grad()
def _outcome_stream(model, data, cfg, bill_margin_cpu, topic_ix_cpu):
    model.eval()
    N_bill = data["bill"].num_nodes
    per_bill = [None]*N_bill

    bill_nei = make_num_neighbors_covering_all(
        data,
        {
            ("bill_version","is_version","bill"):[8,8,8],
            ("bill_version","read","committee"):[6,6,6],
            ("legislator_term","voted_on","bill_version"):[24,16,8],
            ("topic","has","bill"):[12,8,0],
            ("legislator_term","wrote","bill_version"):[6,6,0],
        },
        hops=3, default=0, mirror_revs=True
    )

    loader = NeighborLoader(
        data,
        input_nodes=("bill", torch.arange(N_bill)),
        num_neighbors=bill_nei,
        batch_size=1536,
        shuffle=False,
        num_workers=0
    )

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=False)
        h = model.encode(batch)

        bill_nid_cpu = batch["bill"].n_id.cpu()
        margin = bill_margin_cpu.index_select(0, bill_nid_cpu).to(DEVICE).view(-1,1)
        topic_ix_local = topic_ix_cpu.index_select(0, bill_nid_cpu).to(DEVICE)

        logits = model.outcome_forward(batch, h, margin, topic_ix_local)
        probs = F.softmax(logits, -1).detach().cpu()

        for i, g in enumerate(bill_nid_cpu.tolist()):
            per_bill[g] = {
                "bill_id": int(g),
                "P(pass)": float(probs[i,2]),
                "P(veto)": float(probs[i,1]),
                "P(fail)": float(probs[i,0]),
                "expected_margin": float(probs[i,2] - probs[i,0]),
                "pivotal_actors": [],
                "committee_bottlenecks": []
            }

        del batch, h, logits, probs, margin, topic_ix_local
        try: import torch.mps as _mps; _mps.empty_cache()
        except: pass

    for i in range(N_bill):
        if per_bill[i] is None:
            per_bill[i] = {
                "bill_id": int(i),
                "P(pass)": 0.0, "P(veto)": 0.0, "P(fail)": 1.0,
                "expected_margin": -1.0,
                "pivotal_actors": [], "committee_bottlenecks": []
            }
    return per_bill

def _all_neighbors(g, hops=3):
    out={('*','*','*'):[-1]*hops}
    for et in g.edge_types: out[et]=[-1]*hops
    return out

def _to_dev(batch, dev, gbv2b=None):
    batch = batch.to(dev, non_blocking=False)
    for et in batch.edge_types: batch[et].edge_index = batch[et].edge_index.long()
    if gbv2b is not None: batch._global_bv2bill = gbv2b.to(dev)
    return batch

def _safe_topic_ix(topic_ix_cpu, bill_global_cpu):
    m = bill_global_cpu.ge(0) & bill_global_cpu.lt(topic_ix_cpu.size(0))
    out = torch.zeros_like(bill_global_cpu)
    out[m] = topic_ix_cpu.index_select(0, bill_global_cpu[m])
    return out

def _embed_all(model, data, device, bsz=4096):
    model.eval()
    out={}
    nei=_all_neighbors(data, hops=model.backbone.layers)
    for nt in data.node_types:
        N = data[nt].num_nodes
        store = torch.zeros(N, model.cfg.d, device='cpu')
        loader = NeighborLoader(data, input_nodes=(nt, torch.arange(N)), num_neighbors=nei, batch_size=bsz, shuffle=False, num_workers=0, pin_memory=False)
        for b in loader:
            b=_to_dev(b, device)
            h = model.encode(b)
            if nt in h: store[b[nt].n_id.cpu()] = h[nt].detach().cpu()
        out[nt]=store
    return out

@torch.no_grad()
def build_outputs_full(model, data, cfg, trainer):
    device = next(model.parameters()).device
    topic_ix_cpu = data["bill"].topic_ix.cpu()
    gbv2b_cpu = trainer.global_bv2bill.cpu()
    nei_vote=_all_neighbors(data, hops=cfg.layers)
    nei_bill=_all_neighbors(data, hops=cfg.layers)
    nei_gate=_all_neighbors(data, hops=cfg.layers)

    N_bill = data["bill"].num_nodes
    N_lt = data["legislator_term"].num_nodes
    N_c = data["committee"].num_nodes
    T = int(topic_ix_cpu.max().item()+1) if topic_ix_cpu.numel()>0 else 0

    bill_margin = torch.zeros(N_bill, dtype=torch.float32)
    bill_yes = torch.zeros(N_bill, dtype=torch.float32)
    bill_no  = torch.zeros(N_bill, dtype=torch.float32)
    e_lt=[]; e_bill=[]; e_topic=[]; e_pyes=[]; e_pno=[]; e_mask=[]

    vote_loader = NeighborLoader(data, input_nodes=("legislator_term", torch.arange(N_lt)), num_neighbors=nei_vote, batch_size=max(512, cfg.vote_bsz), shuffle=False, num_workers=0)
    for batch in vote_loader:
        batch=_to_dev(batch, device, gbv2b_cpu)
        h = model.encode(batch)
        e = batch[("legislator_term","voted_on","bill_version")]
        if e is None or e.edge_index.numel()==0: continue
        lt_i, bv_i = e.edge_index
        if ("bill_version","is_version","bill") in batch.edge_types and batch[("bill_version","is_version","bill")].edge_index.numel()>0:
            bill_loc = batch[("bill_version","is_version","bill")].edge_index[1][bv_i]
        else:
            bv_glob = batch["bill_version"].n_id[bv_i].cpu()
            bill_glob = gbv2b_cpu.index_select(0, bv_glob)
            bmap = {int(g):i for i,g in enumerate(batch["bill"].n_id.cpu().tolist())} if "bill" in batch.node_types else {}
            bill_loc = torch.tensor([bmap.get(int(g), -1) for g in bill_glob.tolist()], device=device)
        m = bill_loc.ge(0)
        if not m.any(): continue
        lt_i = lt_i[m]; bv_i = bv_i[m]; bill_loc = bill_loc[m]
        bv_glob = batch["bill_version"].n_id[bv_i].cpu()
        bill_glob = gbv2b_cpu.index_select(0, bv_glob)
        topic_i = _safe_topic_ix(topic_ix_cpu, bill_glob).clamp(min=0).to(device)
        lt_h = h["legislator_term"].index_select(0, lt_i)
        bill_h = h["bill"].index_select(0, bill_loc)
        topic_h = h["topic"].index_select(0, topic_i)
        t2v = {}
        if ("legislator_term","voted_on","bill_version") in batch.edge_types and getattr(batch[("legislator_term","voted_on","bill_version")], "edge_attr", None) is not None:
            t2v = {("legislator_term","voted_on","bill_version"): model.t2v(batch[("legislator_term","voted_on","bill_version")].edge_attr[:,-1].float().to(device))}
        ctx = model.metapath(batch, h, lt_i, bv_i, topic_ix_cpu, t2v, None)
        logits = model.vote_head(lt_h, bill_h, topic_h, ctx)
        p = F.softmax(logits, -1)
        yes, no = p[:,2], p[:,0]
        mv = model.mask_net(lt_h, topic_h, training=False)
        part_yes = scatter_add(yes, bill_loc, dim=0, dim_size=h["bill"].size(0))
        part_no  = scatter_add(no , bill_loc, dim=0, dim_size=h["bill"].size(0))
        bill_margin.index_add_(0, batch["bill"].n_id.cpu(), (part_yes - part_no).detach().cpu())
        bill_yes.index_add_(0, batch["bill"].n_id.cpu(), part_yes.detach().cpu())
        bill_no.index_add_(0,  batch["bill"].n_id.cpu(), part_no.detach().cpu())
        e_lt.append(batch["legislator_term"].n_id[lt_i].cpu())
        e_bill.append(bill_glob.cpu())
        e_topic.append(topic_i.detach().cpu())
        e_pyes.append(yes.detach().cpu())
        e_pno.append(no.detach().cpu())
        e_mask.append(mv.detach().cpu())

    if len(e_lt)==0:
        edge_rows = {"lt": torch.zeros(0, dtype=torch.long), "bill": torch.zeros(0, dtype=torch.long), "topic": torch.zeros(0, dtype=torch.long), "p_yes": torch.zeros(0), "p_no": torch.zeros(0), "mask": torch.zeros(0)}
    else:
        edge_rows = {"lt": torch.cat(e_lt,0), "bill": torch.cat(e_bill,0), "topic": torch.cat(e_topic,0), "p_yes": torch.cat(e_pyes,0), "p_no": torch.cat(e_pno,0), "mask": torch.cat(e_mask,0)}

    comm_adv_sum = torch.zeros(N_c)
    comm_adv_cnt = torch.zeros(N_c)
    bill_bottlenecks = {}
    gate_loader = NeighborLoader(data, input_nodes=("committee", torch.arange(N_c)), num_neighbors=nei_gate, batch_size=max(512, cfg.gate_bsz), shuffle=False, num_workers=0)
    for batch in gate_loader:
        batch=_to_dev(batch, device, gbv2b_cpu)
        h = model.encode(batch)
        rd = batch.get(("bill_version","read","committee"), None)
        if rd is None or rd.edge_index.numel()==0: continue
        bv_i, c_i = rd.edge_index
        bv_glob = batch["bill_version"].n_id[bv_i].cpu()
        bill_glob = gbv2b_cpu.index_select(0, bv_glob)
        bmap = {int(g):i for i,g in enumerate(batch["bill"].n_id.cpu().tolist())} if "bill" in batch.node_types else {}
        bill_loc = torch.tensor([bmap.get(int(g), -1) for g in bill_glob.tolist()], device=device)
        m = bill_loc.ge(0)
        if not m.any(): continue
        bv_i=bv_i[m]; c_i=c_i[m]; bill_loc=bill_loc[m]; bill_glob=bill_glob[m]
        t_i = _safe_topic_ix(topic_ix_cpu, bill_glob).clamp(min=0).to(device)
        logits = model.gate_head(h["committee"].index_select(0, c_i), h["bill"].index_select(0, bill_loc), h["topic"].index_select(0, t_i))
        p = F.softmax(logits, -1).max(-1).values
        comm_global = batch["committee"].n_id[c_i].cpu()
        comm_adv_sum.index_add_(0, comm_global, p.detach().cpu())
        comm_adv_cnt.index_add_(0, comm_global, torch.ones_like(p.detach().cpu()))
        for b,c,v in zip(bill_glob.tolist(), comm_global.tolist(), p.detach().cpu().tolist()):
            bill_bottlenecks.setdefault(b, []).append({"committee_id": int(c), "p_gate": float(v)})
    comm_adv_cnt = comm_adv_cnt.clamp_min(1)
    gate_index = (comm_adv_sum/comm_adv_cnt)

    per_bill = [None]*N_bill
    bill_loader = NeighborLoader(data, input_nodes=("bill", torch.arange(N_bill)), num_neighbors=nei_bill, batch_size=max(124, cfg.bill_bsz), shuffle=False, num_workers=0)
    for batch in bill_loader:
        batch=_to_dev(batch, device, gbv2b_cpu)
        h = model.encode(batch)
        b_nid = batch["bill"].n_id.cpu()
        margin = bill_margin.index_select(0, b_nid).to(device).view(-1,1)
        tloc = topic_ix_cpu.index_select(0, b_nid).to(device)
        logits = model.outcome_forward(batch, h, margin, tloc)
        probs = F.softmax(logits, -1).detach().cpu()
        for i,g in enumerate(b_nid.tolist()):
            per_bill[g] = {
                "bill_id": int(g),
                "p_pass": float(probs[i,2]),
                "p_veto": float(probs[i,1]),
                "p_fail": float(probs[i,0]),
                "exp_margin": float(probs[i,2]-probs[i,0])
            }
    for i in range(N_bill):
        if per_bill[i] is None:
            per_bill[i] = {"bill_id": int(i), "p_pass": 0.0, "p_veto": 0.0, "p_fail": 1.0, "exp_margin": -1.0}

    lt_glob = edge_rows["lt"]; bill_glob = edge_rows["bill"]; p_yes = edge_rows["p_yes"]; p_no = edge_rows["p_no"]; mask=edge_rows["mask"]
    contrib = (p_yes - p_no)
    k = 5
    piv = {}
    if bill_glob.numel()>0:
        for i in range(bill_glob.numel()):
            b = int(bill_glob[i].item()); a = int(lt_glob[i].item()); v = float(contrib[i].item())
            piv.setdefault(b, {})
            piv[b][a] = piv[b].get(a, 0.0) + v
    for b, d in piv.items():
        top = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
        per_bill[b]["pivotal_actors"] = [{"actor_id": aid, "score": float(s)} for aid, s in top]
    for b, lst in bill_bottlenecks.items():
        per_bill[b]["committee_bottlenecks"] = lst

    stance_lbl = trainer.stance_lbl if trainer is not None else torch.full((N_lt, T), float("nan"))
    stance_w   = trainer.stance_w   if trainer is not None else torch.ones(N_lt, T)
    topic_prev = data["topic"].prev if hasattr(data["topic"], "prev") else torch.ones(T)/max(T,1)

    lt_topic_eng = torch.zeros(N_lt, T)
    lt_topic_infl = torch.zeros(N_lt, T)
    lt_support_votes = torch.zeros(N_lt, dtype=torch.long)

    if bill_glob.numel()>0:
        for i in range(bill_glob.numel()):
            lt = int(lt_glob[i].item()); b = int(bill_glob[i].item())
            if b<0 or b>=topic_ix_cpu.numel(): continue
            t = int(topic_ix_cpu[b].item())
            if t<0 or t>=T: continue
            lt_topic_eng[lt, t] += 1.0
            delta = float((p_yes[i]-p_no[i]).item()) * float(mask[i].item())
            lt_topic_infl[lt, t] += delta
            lt_support_votes[lt] += 1

    rows=[]
    for lt in range(N_lt):
        for t in range(T):
            s_val = stance_lbl[lt, t]
            s = float(s_val.item()) if torch.isfinite(s_val) else 0.0
            w = float(stance_w[lt, t].item()) if torch.isfinite(s_val) else 0.0
            infl = float(lt_topic_infl[lt, t].item())
            eng  = float(lt_topic_eng[lt, t].item())
            ci = 0.1 / math.sqrt(max(1.0, eng))
            rows.append({
                "actor_id": lt, "actor_type": "legislator_term", "topic_id": t,
                "stance": s, "stance_ci_lo": s-ci, "stance_ci_hi": s+ci,
                "influence_delta_mean": infl, "influence_ci_lo": infl-ci, "influence_ci_hi": infl+ci,
                "engagement": eng, "certainty": min(1.0, w)
            })
    actor_topic_df = pd.DataFrame(rows)

    actor_overall_vals=[]
    for lt in range(N_lt):
        weights = lt_topic_eng[lt].numpy()
        W = float(weights.sum())
        if W>0: weights = weights/W
        else: weights = np.full(T, 1.0/max(1,T))
        overall = 0.0
        for t in range(T):
            overall += weights[t] * float(lt_topic_infl[lt, t].item()) * float(topic_prev[t].item())
        actor_overall_vals.append({
            "actor_id": lt, "actor_type": "legislator_term",
            "overall_influence": overall, "ci_lo": overall-0.1, "ci_hi": overall+0.1
        })
    actor_overall_df = pd.DataFrame(actor_overall_vals)

    committee_overall_df = pd.DataFrame({
        "committee_id": np.arange(N_c, dtype=int),
        "overall_influence": gate_index.numpy().astype(float),
        "ci_lo": (gate_index-0.05).numpy().astype(float),
        "ci_hi": (gate_index+0.05).numpy().astype(float),
        "gate_index": gate_index.numpy().astype(float)
    })

    per_bill_df = pd.DataFrame(per_bill)
    if "pivotal_actors" not in per_bill_df.columns: per_bill_df["pivotal_actors"] = [[] for _ in range(per_bill_df.shape[0])]
    if "committee_bottlenecks" not in per_bill_df.columns: per_bill_df["committee_bottlenecks"] = [[] for _ in range(per_bill_df.shape[0])]

    want_cols_pb = ["bill_id","p_pass","p_veto","p_fail","exp_margin","pivotal_actors","committee_bottlenecks"]
    per_bill_df = per_bill_df[want_cols_pb]

    actor_topic_df = actor_topic_df.sort_values(["actor_id","topic_id"]).reset_index(drop=True)
    actor_overall_df = actor_overall_df.sort_values(["actor_id"]).reset_index(drop=True)
    committee_overall_df = committee_overall_df.sort_values(["committee_id"]).reset_index(drop=True)

    return {
        "per_bill": per_bill_df,
        "actor_topic": actor_topic_df,
        "actor_overall": actor_overall_df,
        "committee_overall": committee_overall_df
    }

# ------------------------------- Runner --------------------------------------
def run_training(epochs, graph_path="data4.pt"):
    data = torch.load(graph_path, weights_only=False)
    tr = Trainer(data, CFG())
    best_loss, best_model = float("inf"), None
    for ep in tqdm(range(epochs)):
        stats = tr.train_epoch()
        evals = tr.evaluate()
        print(f"epoch {ep} :: ", {k: round(v,4) for k,v in stats.items()}, "| eval:", evals)
        gc.collect()
        loss = sum([v for k,v in stats.items() if k.startswith("loss_")])
        if loss < best_loss:
            best_loss = loss
            best_model = tr.model.state_dict()
    return tr, best_model

RUN_TYPE = 'train' # 'train' or 'output'
if __name__=="__main__":
    if RUN_TYPE=='train':
        trainer, best_model = run_training(epochs=2)
        torch.save(best_model, "legnn4_best_model.pt")

    elif RUN_TYPE=='output':
        data = torch.load("data4.pt", weights_only=False)
        cfg = CFG()
        topic_builder = TopicBuilder(expected=cfg.topics_expected)
        base = data.clone()
        T, topic_ix = topic_builder(base)
        base["topic"].x = torch.eye(T)
        data = base
        best_model = torch.load("legnn4_best_model.pt")
        tr = Trainer(data, CFG())
        tr.model.load_state_dict(best_model)
        outputs = build_outputs_full(tr.model, data, tr.cfg, trainer=tr)
        torch.save(outputs, "legnn4_outputs.pt")


