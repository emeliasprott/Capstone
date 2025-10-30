import os, math, json, gc, random, warnings, torch, numpy as np, pandas as pd
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

def _dev():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

@dataclass
class CFG:
    seed=40
    d=128
    heads=2
    layers=2
    drop=0.15
    lr=1e-3
    wd=1e-4
    epochs=10
    bsz=4096
    neigh_budgets={"bill_version":[50,25],"bill":[70,30],"legislator_term":[24,16],"legislator":[24,16],"committee":[24,16],"lobby_firm":[24,16],"donor":[24,16]}
    time_decay=0.95
    contrastive_tau=0.07
    lambda_edge=1.0
    lambda_outcome=0.5
    lambda_contrast=0.25
    lambda_temporal=0.1
    lambda_margin=0.5
    save_dir="outs"
    topic_dim=64
    topic_count=77
    topic_pad=0
    eval_snapshots=None
    max_snapshots=12
    pe_k=8
    margin=0.5
    neg_ratio=1
    topk_topics=10
    pe_steps=12
    pe_tau=0.5
    msg_chunk_edges=200_000

DEVICE=_dev()
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

MANUAL_EDGE_TCOLS={
 ('legislator_term','voted_on','bill_version'):[-1],
 ('committee','rev_read','bill_version'):[-1],
 ('donor','donated_to','legislator_term'):[-1],
 ('lobby_firm','lobbied','committee'):[-1],
 ('lobby_firm','lobbied','legislator_term'):[-1],
 ('bill_version','read','committee'):[-1],
}

def empty_cache_mps():
    if hasattr(torch,"mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass

def load_hetero(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def ensure_bidirectional(data: HeteroData):
    for (s,r,t) in list(data.edge_types):
        e = data[(s,r,t)]
        if e.edge_index.numel()==0:
            rev = (t, r+"_rev", s)
            if rev not in data.edge_types:
                data[rev].edge_index = e.edge_index.flip(0)
            continue
        src, dst = e.edge_index
        rev = (t, r+"_rev", s)
        if rev not in data.edge_types:
            data[rev].edge_index = torch.stack([dst, src], dim=0)
            if "edge_attr" in e and e.edge_attr is not None and e.edge_attr.size(0)==src.size(0):
                data[rev].edge_attr = e.edge_attr.clone()

def normalize_node_features(data):
    for nt in data.node_types:
        if "x" in data[nt]:
            x = data[nt].x
            if x is None or x.numel()==0:
                continue
            if x.dtype not in (torch.float16, torch.float32, torch.float64):
                continue
            x = x.clone()
            mask = torch.isfinite(x)
            if not mask.all():
                x[~mask] = 0.0
            m = x.mean(0, keepdim=True)
            v = x.var(0, unbiased=False, keepdim=True)
            v = torch.clamp(v, min=1e-8)
            data[nt].x = (x - m) / torch.sqrt(v)

def get_edge_time_attr(data, etype, default_t=0.0):
    try:
        e=data[etype]
        E=e.edge_index.size(1)
        if "edge_attr" in e and e.edge_attr is not None and e.edge_attr.size(0)==E:
            cols=MANUAL_EDGE_TCOLS.get(etype, None)
            if cols: return e.edge_attr[:, cols[0]].float()
        for key in ("time","timestamp","date"):
            if key in e:
                v=e[key]
                if isinstance(v, torch.Tensor) and v.numel()==E:
                    return v.float()
    except:
        pass
    return torch.full((data[etype].edge_index.size(1),), float(default_t))

def build_time_slices(data):
    caps_per_edge=200000
    cap_total=1000000
    ts=[]
    for et in data.edge_types:
        t=get_edge_time_attr(data, et)
        if t.numel()==0:
            continue
        t=t.detach().cpu().float()
        n=t.numel()
        if n>caps_per_edge:
            idx=torch.randint(0,n,(caps_per_edge,))
            t=t[idx]
        ts.append(t)
    if len(ts)==0:
        return [None]
    all_t=torch.cat(ts)
    if all_t.numel()>cap_total:
        idx=torch.randint(0, all_t.numel(), (cap_total,))
        all_t=all_t[idx]
    arr=all_t.numpy()
    qs=np.quantile(arr, np.linspace(0.0,1.0,CFG.max_snapshots+1))
    edges_by_snap=[(float(qs[s]), float(qs[s+1])) for s in range(CFG.max_snapshots)]
    return edges_by_snap

def filter_graph_by_time(data, time_window):
    if time_window is None:
        return data
    lo,hi=time_window
    out=HeteroData()
    for nt in data.node_types:
        out[nt].num_nodes=data[nt].num_nodes
        for f in data[nt].keys(): out[nt][f]=data[nt][f]
    for et in data.edge_types:
        eidx=data[et].edge_index
        if eidx.numel()==0:
            out[et].edge_index=eidx
            for f in data[et].keys():
                if f!="edge_index":
                    out[et][f]=data[et][f]
            continue
        t=get_edge_time_attr(data, et)
        if t.numel()==0:
            out[et].edge_index=eidx
            for f in data[et].keys():
                if f!="edge_index":
                    out[et][f]=data[et][f]
            continue
        mask=(t>=lo)&(t<=hi)
        keep=torch.where(mask)[0]
        out[et].edge_index=eidx[:,keep]
        for f in data[et].keys():
            if f=="edge_index":
                continue
            val=data[et][f]
            if isinstance(val, torch.Tensor) and val.size(0)==eidx.size(1):
                out[et][f]=val[keep]
            else:
                out[et][f]=val
    return out

def per_type_laplacian_pe(data):
    pe={}
    for nt in data.node_types:
        n=data[nt].num_nodes
        if n<=1:
            pe[nt]=torch.zeros(n,CFG.pe_k)
            continue
        rows=[]; cols=[]
        for (s,r,t) in data.edge_types:
            e=data[(s,r,t)].edge_index
            if e.numel()==0:
                continue
            if s==nt: rows.append(e[0]); cols.append(e[1])
            if t==nt: rows.append(e[1]); cols.append(e[0])
        if len(rows)==0:
            pe[nt]=torch.zeros(n,CFG.pe_k)
            continue
        row=torch.cat(rows).long(); col=torch.cat(cols).long()
        mask=(row>=0)&(row<n)&(col>=0)&(col<n)
        row=row[mask]; col=col[mask]
        if row.numel()==0:
            pe[nt]=torch.zeros(n,CFG.pe_k)
            continue
        idx=torch.stack([row, col])
        val=torch.ones(idx.size(1))
        A=torch.sparse_coo_tensor(idx, val, (n,n)).coalesce()
        deg=torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
        dinv=deg.pow(-0.5)
        def Lx(x):
            y=torch.sparse.mm(A, x)
            y = y*dinv.unsqueeze(-1)
            x2= x*dinv.unsqueeze(-1)
            return x2 - y
        X=torch.randn(n, CFG.pe_k)
        for s in range(CFG.pe_steps):
            X = X - CFG.pe_tau*Lx(X)
            if (s+1)%3==0:
                q,_=torch.linalg.qr(X, mode='reduced')
                X=q
        q,_=torch.linalg.qr(X, mode='reduced')
        pe[nt]=q[:, :CFG.pe_k].contiguous()
    return pe

class TypeLinear(nn.Module):
    def __init__(self, in_dims, d, drop, pe_dims):
        super().__init__()
        self.ts = list(in_dims.keys())
        self.pe_dims = pe_dims
        self.lins = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(in_dims[t] + pe_dims.get(t, 0), d),
                nn.ReLU(),
                nn.Dropout(drop)
            )
            for t in self.ts
        })

    def forward(self, xdict, pedict):
        out = {}
        for t in self.ts:
            x = xdict[t]["x"]
            pe = pedict.get(t, None)

            if pe is not None:
                if pe.size(0) != x.size(0):
                    pe = torch.zeros(x.size(0), pe.size(1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pe.to(x.device, dtype=x.dtype)], dim=-1)

            lin0 = self.lins[t][0]
            expected_in = lin0.in_features
            cur_in = x.size(1)

            if cur_in < expected_in:
                pad = torch.zeros(x.size(0), expected_in - cur_in, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
            elif cur_in > expected_in:
                x = x[:, :expected_in]

            if x.size(0) == 0:
                out[t] = torch.zeros(0, lin0.out_features, device=x.device, dtype=x.dtype)
            else:
                out[t] = self.lins[t](x)
        return out


class RelSAGEConv(nn.Module):
    def __init__(self, in_src, in_dst, out, drop, msg_chunk_edges=200_000):
        super().__init__()
        self.lin_src = nn.Linear(in_src, out)
        self.lin_dst = nn.Linear(in_dst, out)
        self.lin_m = nn.Linear(out, out)
        self.drop = nn.Dropout(drop)
        self.msg_chunk_edges = msg_chunk_edges

    def forward(self, x_src, x_dst, edge_index):
        if edge_index.numel() == 0 or x_dst.size(0) == 0:
            return F.relu(self.lin_dst(x_dst))

        src, dst = edge_index
        out = torch.zeros(x_dst.size(0), self.lin_src.out_features, device=x_dst.device, dtype=x_dst.dtype)

        deg = torch.bincount(dst, minlength=x_dst.size(0)).clamp_min(1).unsqueeze(-1).to(out.dtype)
        E = src.size(0)
        step = self.msg_chunk_edges
        for start in range(0, E, step):
            end = min(start + step, E)
            s = src[start:end]
            d = dst[start:end]
            m = self.lin_src(x_src[s])
            out.index_add_(0, d, m)

        out = out / deg
        out = self.lin_m(out) + self.lin_dst(x_dst)
        return self.drop(F.relu(out))


class MetaPathGater(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(3*d,d),nn.ReLU(),nn.Linear(d,1))
    def forward(self, a, bv, b):
        g=torch.sigmoid(self.gate(torch.cat([a,bv,b],dim=-1)))
        return g

class TopicAttention(nn.Module):
    def __init__(self, d, tdim):
        super().__init__()
        self.q=nn.Linear(tdim,d)
        self.k=nn.Linear(d,d)
        self.v=nn.Linear(d,d)
        self.scale=math.sqrt(d)
    def forward(self, h, tvec):
        q=self.q(tvec)
        k=self.k(h)
        v=self.v(h)
        s=(q*k).sum(-1)/self.scale
        a=torch.sigmoid(s).unsqueeze(-1)
        o=a*v
        return o, a.squeeze(-1)

class RelationalSAGE(nn.Module):
    def __init__(self, metadata, d, layers, drop, topic_dim, topic_count, actor_types):
        super().__init__()
        self.node_types, self.edge_types=metadata
        self.d=d
        self.layers=layers
        self.drop=drop
        self.actor_types=actor_types
        self.relations=nn.ModuleList()
        for _ in range(layers):
            rel_layer=nn.ModuleDict()
            for (s,r,t) in self.edge_types:
                rel_layer[f"{s}|{r}|{t}"] = RelSAGEConv(d, d, d, drop, msg_chunk_edges=getattr(CFG, "msg_chunk_edges", 200_000))
            self.relations.append(rel_layer)
        self.rel_attn=nn.ParameterDict({f"{s}|{r}|{t}":nn.Parameter(torch.zeros(1)) for (s,r,t) in self.edge_types})
        self.metapath_gate=MetaPathGater(d)
        self.topic_table=nn.Embedding(topic_count+1, topic_dim)
        nn.init.xavier_uniform_(self.topic_table.weight)
        self.topic_proj=nn.Linear(topic_dim,d)
        self.topic_attn=TopicAttention(d, d)
        self.cached_alphas={}
    def forward(self, h, adict, topic_ids=None, topic_emb_override=None):
        self.cached_alphas={}
        T=None
        if topic_ids is not None:
            T=self.topic_proj(self.topic_table(topic_ids))
        elif topic_emb_override is not None:
            T=self.topic_proj(topic_emb_override)
        for l in range(self.layers):
            new={k:torch.zeros_like(v) for k,v in h.items()}
            rel_layer=self.relations[l]
            for (s,r,t) in self.edge_types:
                key=f"{s}|{r}|{t}"
                if key not in rel_layer or (s,r,t) not in adict:
                    continue
                m=rel_layer[key](h[s], h[t], adict[(s,r,t)])
                a=torch.sigmoid(self.rel_attn[key])
                new[t]=new[t]+a*m
            for nt in new: new[nt]=F.relu(new[nt])
            if T is not None:
                for at in self.actor_types:
                    if at in new:
                        ta, alpha=self.topic_attn(new[at], T[:new[at].size(0)])
                        new[at]=0.5*new[at]+0.5*ta
                        self.cached_alphas[at]=alpha.detach()
            h=new
        if ("legislator_term" in h) and ("bill_version" in h) and ("bill" in h) and ("bill_version","is_version","bill") in adict:
            e=adict[("bill_version","is_version","bill")]
            bv=h["bill_version"]; b=h["bill"]
            agg=torch.zeros_like(bv)
            if e.numel()>0:
                agg.index_add_(0, e[0], b[e[1]])
                deg=torch.bincount(e[0], minlength=bv.size(0)).clamp_min(1).unsqueeze(-1)
                agg=agg/deg
                g=self.metapath_gate(bv, bv, agg)
                h["bill_version"]=g*agg+(1-g)*bv
        return h

class InfluenceHead(nn.Module):
    def __init__(self, d, topic_dim, topic_count):
        super().__init__()
        self.actor_proj=nn.Linear(d,d)
        self.topic_table=nn.Embedding(topic_count+1, topic_dim)
        nn.init.xavier_uniform_(self.topic_table.weight)
        self.topic_proj=nn.Linear(topic_dim,d)
        self.scale=nn.Parameter(torch.tensor(1.0))
    def forward(self, actor_emb, topic_ids, topic_emb_optional=None):
        a=self.actor_proj(actor_emb)
        t=self.topic_proj(self.topic_table(topic_ids)) if topic_emb_optional is None else self.topic_proj(topic_emb_optional)
        return (a*t).sum(-1)*self.scale

class OutcomeHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d,3))
    def forward(self, bill_emb):
        return self.mlp(bill_emb)

class EdgePredHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d,1))
    def forward(self, src, dst):
        return self.scorer(torch.cat([src,dst],dim=-1)).squeeze(-1)

class ContrastiveProj(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.p=nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d,d))
    def forward(self, x):
        return F.normalize(self.p(x), dim=-1)

class DecompHead(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, rel_attn, per_rel_msgs, actor_idx):
        contrib={}
        for k,v in per_rel_msgs.items():
            if v is None:
                continue
            a=torch.sigmoid(rel_attn[k])
            if v.size(0)<=actor_idx.max().item():
                continue
            contrib[k]=a*v[actor_idx]
        return contrib

@torch.no_grad()
def infer_embeddings_batched(model, data, bsz=8192):
    model.eval()
    embs = {nt: torch.zeros(data[nt].num_nodes, model.d, device=DEVICE) for nt in data.node_types}

    for nt in data.node_types:
        input_nodes = (nt, torch.arange(data[nt].num_nodes))
        num_neighbors = {}
        depth = getattr(CFG, "layers", 2)
        for et in data.edge_types:
            num_neighbors[et] = [-1] * depth

        loader = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=num_neighbors,
            batch_size=bsz,
            shuffle=False
        )

        for batch in tqdm(loader, desc=f"infer {nt}", leave=False):
            batch = batch.to(DEVICE)
            h = model(batch)
            gidx = batch[nt].n_id.to(DEVICE)
            embs[nt][gidx] = h[nt]

            del h, batch
            gc.collect()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    for nt in embs:
        embs[nt] = embs[nt].detach().cpu()
    return embs


class Model(nn.Module):
    def __init__(self, data, d, layers, drop, topic_dim, topic_count, actor_types, pe):
        super().__init__()
        in_dims={}
        pe_dims={}
        for nt in data.node_types:
            in_dims[nt]=data[nt].x.size(1) if "x" in data[nt] else d
            pe_dims[nt]=pe[nt].size(1) if (pe is not None and nt in pe) else 0
        self.pe_dims=pe_dims
        self.enc=TypeLinear(in_dims, d, drop, pe_dims)
        self.backbone=RelationalSAGE(data.metadata(), d, layers, drop, topic_dim, topic_count, actor_types)
        self.infl=InfluenceHead(d, topic_dim, topic_count)
        self.outcome=OutcomeHead(d)
        self.edgepred=EdgePredHead(d)
        self.cproj=ContrastiveProj(d)
        self.d=d
        self.actor_types=actor_types
        self.pe=pe
    def forward(self, batch, topic_ids=None, topic_emb_override=None, capture_msgs=False):
        xdict = {}
        pedict = {}
        for nt in batch.node_types:
            if "x" in batch[nt]:
                x_nt = batch[nt].x.to(DEVICE)
            else:
                x_nt = torch.zeros(batch[nt].num_nodes, self.d, device=DEVICE)
            xdict[nt] = {"x": x_nt}
            exp_pe = self.pe_dims.get(nt, 0)
            if exp_pe > 0:
                if (self.pe is not None) and (nt in self.pe) and (self.pe[nt] is not None) and (self.pe[nt].numel() > 0):
                    if hasattr(batch[nt], "n_id"):
                        idx = batch[nt].n_id.to(torch.long, non_blocking=True)
                        pe_nt = self.pe[nt][idx.cpu()].to(DEVICE)
                    else:
                        pe_nt = self.pe[nt].to(DEVICE)
                    if pe_nt.size(0) != x_nt.size(0):
                        pe_nt = torch.zeros(x_nt.size(0), exp_pe, device=DEVICE, dtype=x_nt.dtype)
                else:
                    pe_nt = torch.zeros(x_nt.size(0), exp_pe, device=DEVICE, dtype=x_nt.dtype)
                pedict[nt] = pe_nt

        h0 = self.enc(xdict, pedict)

        adict = {et: batch[et].edge_index.to(DEVICE) for et in batch.edge_types}
        h = self.backbone(h0, adict, topic_ids, topic_emb_override)

        if capture_msgs:
            per_rel = {}
            for (s, r, t) in batch.edge_types:
                key = f"{s}|{r}|{t}"
                if t in h and s in h:
                    per_rel[key] = h[t]
            return h, self.backbone.rel_attn, per_rel
        return h


def outcome_loss(model, h, batch):
    if "bill" not in batch.node_types or "outcome" not in batch["bill"]:
        return torch.tensor(0.0, device=DEVICE)
    y=batch["bill"].outcome.to(DEVICE).long()
    logits=model.outcome(h["bill"])
    return F.cross_entropy(logits, y, ignore_index=-1)

def build_negatives(eidx, n_src, n_dst, num_neg):
    src=eidx[0]; dst=eidx[1]
    ns=[]
    for _ in range(num_neg):
        ridx=torch.randint(0,src.size(0),(src.size(0),), device=src.device)
        rs=src[ridx]
        rd=torch.randint(0,n_dst,(dst.size(0),), device=dst.device)
        ns.append((rs, rd))
    return ns

def edge_margin_loss(model, h, batch, et):
    if et not in batch.edge_types:
        return torch.tensor(0.0, device=DEVICE)
    eidx=batch[et].edge_index.to(DEVICE)
    if eidx.numel()==0:
        return torch.tensor(0.0, device=DEVICE)
    s_type, _, t_type = et
    pos=model.edgepred(h[s_type][eidx[0]], h[t_type][eidx[1]])
    n_src=h[s_type].size(0); n_dst=h[t_type].size(0)
    negs=build_negatives(eidx, n_src, n_dst, CFG.neg_ratio)
    loss=0.0; c=0
    for rs, rd in negs:
        neg=model.edgepred(h[s_type][rs], h[t_type][rd])
        loss=loss+F.relu(CFG.margin-(pos-neg)).mean()
        c+=1
    return loss/max(1,c)

def _paired_bill_bv_indices(batch):
    if ("bill_version","is_version","bill") not in batch.edge_types: return None
    e = batch[("bill_version","is_version","bill")].edge_index
    # create per-bill one representative bv (or many); here align by positions safely
    bv=e[0]; b=e[1]
    if bv.numel()==0:
        return None
    # keep up to one bv per bill to avoid duplicates
    uniq_b, first_idx = torch.unique_consecutive(b.sort()[0], return_inverse=False, return_counts=False), None
    # simpler: just return aligned pairs (truncate to equal length)
    n=min(bv.size(0), b.size(0))
    return bv[:n], b[:n]

def contrastive_loss(model, h, batch, tau):
    if "bill" not in h or "bill_version" not in h:
        return torch.tensor(0.0, device=DEVICE)
    pairs = _paired_bill_bv_indices(batch)
    if pairs is None: return torch.tensor(0.0, device=DEVICE)
    bv_idx, b_idx = pairs
    if bv_idx.numel()<2: return torch.tensor(0.0, device=DEVICE)
    a=model.cproj(h["bill_version"][bv_idx.to(DEVICE)])
    b=model.cproj(h["bill"][b_idx.to(DEVICE)])
    n=min(a.size(0), b.size(0), 4096)
    a=a[:n]
    b=b[:n]
    sim = a @ b.t()  # (n,n)
    targets = torch.arange(n, device=DEVICE)
    loss = (F.cross_entropy(sim/tau, targets) + F.cross_entropy(sim.t()/tau, targets)) * 0.5
    return loss

def temporal_smooth_loss(h_prev, h_curr, keys):
    if h_prev is None: return torch.tensor(0.0, device=DEVICE)
    ks=[]
    for k in keys:
        if k in h_prev and k in h_curr and h_prev[k].size(0)>0 and h_curr[k].size(0)>0:
            n=min(h_prev[k].size(0), h_curr[k].size(0))
            ks.append(F.mse_loss(h_curr[k][:n], h_prev[k][:n]))
    if len(ks)==0: return torch.tensor(0.0, device=DEVICE)
    return torch.stack(ks).mean()

def choose_input_nodes(data):
    if "legislator_term" in data.node_types:
        return ("legislator_term", torch.arange(data["legislator_term"].num_nodes))
    if "bill" in data.node_types:
        return ("bill", torch.arange(data["bill"].num_nodes))
    nt=list(data.node_types)[0]
    return (nt, torch.arange(data[nt].num_nodes))

def prepare_num_neighbors(data):
    depth = CFG.layers
    out={}
    for et in data.edge_types:
        dst=et[2]
        budget = CFG.neigh_budgets.get(dst, [10,8])
        if len(budget)<depth:
            budget = budget + [budget[-1]]*(depth-len(budget))
        elif len(budget)>depth:
            budget = budget[:depth]
        out[et]=budget
    return out

def extract_topic_ids_from_batch(batch, max_n):
    if "bill" in batch.node_types and "topic_id" in batch["bill"]:
        tids=batch["bill"].topic_id.to(DEVICE)
        if tids.numel()==0:
            return torch.zeros(max_n, dtype=torch.long, device=DEVICE)
        if tids.size(0)>=max_n:
            return tids[:max_n]
        rep=max_n//tids.size(0)+1
        return tids.repeat(rep)[:max_n]
    return torch.zeros(max_n, dtype=torch.long, device=DEVICE)

def train_one_snapshot(model, data, optimizer, sid):
    model.train()
    input_nodes=choose_input_nodes(data)
    num_neighbors=prepare_num_neighbors(data)
    loader=NeighborLoader(data, input_nodes=input_nodes, num_neighbors=num_neighbors, batch_size=CFG.bsz, shuffle=True)
    prev_h=None
    keys=["legislator_term","committee","donor","lobby_firm","bill","bill_version"]
    total=0.0; steps=0
    pbar=tqdm(loader, desc=f"snapshot {sid} train", leave=False)
    for batch in pbar:
        for nt in batch.node_types:
            if "x" in batch[nt]:
                batch[nt].x=batch[nt].x.to(DEVICE)
            if "topic_id" in batch[nt]:
                batch[nt].topic_id=batch[nt].topic_id.to(DEVICE)
            if "outcome" in batch[nt]:
                batch[nt].outcome=batch[nt].outcome.to(DEVICE)
        batch=batch.to(DEVICE)
        n_actors=sum([(batch[at].num_nodes if at in batch.node_types else 0) for at in model.actor_types])
        topic_ids=extract_topic_ids_from_batch(batch, max_n=max(1,n_actors))
        h=model(batch, topic_ids=topic_ids)
        l_out=outcome_loss(model,h,batch)*CFG.lambda_outcome
        l_edge=0.0
        for et in [("legislator_term","voted_on","bill_version"),("donor","donated_to","legislator_term"),("lobby_firm","lobbied","legislator_term"),("lobby_firm","lobbied","committee"),("legislator_term","wrote","bill_version")]:
            if et in batch.edge_types:
                l_edge=l_edge+edge_margin_loss(model,h,batch,et)
        l_edge=l_edge*CFG.lambda_edge
        l_ctr=contrastive_loss(model,h,batch,CFG.contrastive_tau)*CFG.lambda_contrast
        l_tmp=temporal_smooth_loss(prev_h,h,keys)*CFG.lambda_temporal
        loss=l_out+l_edge+l_ctr+l_tmp
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total+=loss.item(); steps+=1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        prev_h={k:v.detach() for k,v in h.items()}
        del h, batch
        gc.collect(); empty_cache_mps()
    return total/max(1,steps)

def run_train(data_path, save_prefix="leginflu_v3"):
    os.makedirs(CFG.save_dir, exist_ok=True)
    data=load_hetero(data_path)
    ensure_bidirectional(data)
    normalize_node_features(data)
    pe=per_type_laplacian_pe(data)
    slices=build_time_slices(data)
    if not slices:
        slices=[None]
    actor_types=["legislator_term","committee","donor","lobby_firm"]
    model=Model(data, CFG.d, CFG.layers, CFG.drop, CFG.topic_dim, CFG.topic_count, actor_types, pe).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    hist=[]
    snap_ids=list(range(len(slices))) if CFG.eval_snapshots is None else CFG.eval_snapshots
    for sid in tqdm(snap_ids, desc="snapshots"):
        snap=filter_graph_by_time(data, slices[sid] if slices[0] is not None else None)
        for e in tqdm(range(CFG.epochs), desc=f"snapshot {sid} epochs", leave=False):
            loss=train_one_snapshot(model, snap, opt, sid)
            hist.append({"snapshot":sid,"epoch":e,"loss":float(loss)})
            print(f"snapshot {sid} epoch {e} loss {loss:.6f}")
    pd.DataFrame(hist).to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_train_hist.parquet"), index=False)
    full_graph = filter_graph_by_time(data, slices[-1] if slices[0] is not None else None)
    h = infer_embeddings_batched(model, full_graph, bsz=min(8192, CFG.bsz))
    rel_attn = model.backbone.rel_attn
    per_rel = {}
    torch.save({"state_dict":model.state_dict(),"cfg":{k:getattr(CFG,k) for k in CFG.__dict__.keys() if not k.startswith('__')}}, os.path.join(CFG.save_dir, f"{save_prefix}_model.pt"))
    torch.save(h, os.path.join(CFG.save_dir, f"{save_prefix}_embeddings.pt"))
    build_actor_topic_outputs(data, h, save_prefix, model, rel_attn, per_rel)
    build_overall_influence_series(data, h, save_prefix)
    return {"model":os.path.join(CFG.save_dir, f"{save_prefix}_model.pt"),
            "embeddings":os.path.join(CFG.save_dir, f"{save_prefix}_embeddings.pt"),
            "hist":os.path.join(CFG.save_dir, f"{save_prefix}_train_hist.parquet")}

def aggregate_actor_topic_signals(data):
    out={}
    if ("legislator_term","voted_on","bill_version") in data.edge_types:
        eidx=data[("legislator_term","voted_on","bill_version")].edge_index
        vattr=data[("legislator_term","voted_on","bill_version")].edge_attr if "edge_attr" in data[("legislator_term","voted_on","bill_version")] else None
        vote_sign=vattr[:,0].float() if (vattr is not None and vattr.size(1)>0) else torch.zeros(eidx.size(1)).float()
        bv2b=torch.zeros(data["bill_version"].num_nodes, dtype=torch.long)
        if ("bill_version","is_version","bill") in data.edge_types:
            pidx=data[("bill_version","is_version","bill")].edge_index
            bv2b[pidx[0]]=pidx[1]
        b_topic=data["bill"].topic_id if "topic_id" in data["bill"] else torch.zeros(data["bill"].num_nodes, dtype=torch.long)
        b_outcome=data["bill"].outcome if "outcome" in data["bill"] else torch.zeros(data["bill"].num_nodes, dtype=torch.long)
        b_k=b_topic[bv2b[eidx[1]]]
        b_y=b_outcome[bv2b[eidx[1]]].float()
        lt=eidx[0]
        align=torch.sign(vote_sign)*torch.sign(b_y)
        df=pd.DataFrame({"legislator_term":lt.cpu().numpy(),"topic":b_k.cpu().numpy(),"vote":vote_sign.cpu().numpy(),"outcome":b_y.cpu().numpy(),"align":align.cpu().numpy()})
        g=df.groupby(["legislator_term","topic"]).agg(events=("vote","count"), support=("align","sum")).reset_index()
        out["legislator_term"]=g
    for et,at in [(("donor","donated_to","legislator_term"),"donor"), (("lobby_firm","lobbied","legislator_term"),"lobby_firm")]:
        if et in data.edge_types and ("legislator_term","voted_on","bill_version") in data.edge_types:
            eidx_v=data[("legislator_term","voted_on","bill_version")].edge_index
            vattr=data[("legislator_term","voted_on","bill_version")].edge_attr if "edge_attr" in data[("legislator_term","voted_on","bill_version")] else None
            vote_sign=vattr[:,0].float() if (vattr is not None and vattr.size(1)>0) else torch.zeros(eidx_v.size(1)).float()
            bv2b=torch.zeros(data["bill_version"].num_nodes, dtype=torch.long)
            if ("bill_version","is_version","bill") in data.edge_types:
                pidx=data[("bill_version","is_version","bill")].edge_index
                bv2b[pidx[0]]=pidx[1]
            b_topic=data["bill"].topic_id if "topic_id" in data["bill"] else torch.zeros(data["bill"].num_nodes, dtype=torch.long)
            b_outcome=data["bill"].outcome if "outcome" in data["bill"] else torch.zeros(data["bill"].num_nodes, dtype=torch.long)
            lt=eidx_v[0]; bv=eidx_v[1]
            b=bv2b[bv]; k=b_topic[b]; y=b_outcome[b].float(); vs=vote_sign
            # aggregate votes per legislator_term-topic
            df_v=pd.DataFrame({"lt":lt.cpu().numpy(),"topic":k.cpu().numpy(),"align":(torch.sign(vs)*torch.sign(y)).cpu().numpy()})
            g_v=df_v.groupby(["lt","topic"]).agg(events=("align","count"), support=("align","sum")).reset_index().rename(columns={"lt":"legislator_term"})
            eidx=data[et].edge_index
            src=eidx[0].cpu().numpy(); lts=eidx[1].cpu().numpy()
            df_e=pd.DataFrame({"a":src,"legislator_term":lts})
            m=df_e.merge(g_v, on="legislator_term", how="left").fillna(0.0)
            g=m.groupby(["a","topic"]).agg(events=("events","sum"), support=("support","sum")).reset_index().rename(columns={"a":at})
            out[at]=g
    return out

def compute_actor_topic_scores(emb, model):
    scores={}
    te=F.normalize(model.infl.topic_proj(model.infl.topic_table.weight),dim=-1)
    for at in ["legislator_term","committee","donor","lobby_firm"]:
        if at in emb:
            A=F.normalize(emb[at],dim=-1).to(DEVICE)
            S=A@te.T
            scores[at]=S.detach().cpu()
    return scores

def decompose_actor_topic(backbone, rel_attn, per_rel, actor_type, actor_idx, topic_vec):
    decomp={}
    for k,v in per_rel.items():
        if v is None:
            continue
        if actor_type in k:
            a=torch.sigmoid(rel_attn[k]).detach().cpu().item()
            if actor_idx<v.size(0):
                s=float((v[actor_idx].unsqueeze(0)@topic_vec.cpu().unsqueeze(-1)).squeeze().item())
                decomp[k]=a*s
    return decomp

def build_actor_topic_outputs(data, emb, save_prefix, model, rel_attn, per_rel):
    agg=aggregate_actor_topic_signals(data)
    scores=compute_actor_topic_scores(emb, model)
    outs=[]
    for at in ["legislator_term","committee","donor","lobby_firm"]:
        if at not in scores:
            continue
        S=scores[at].numpy()
        n=S.shape[0]
        topk=min(CFG.topk_topics, S.shape[1])
        idx_top=np.argpartition(-S, topk-1, axis=1)[:,:topk]
        rows=[]
        for i in range(n):
            for j in idx_top[i]:
                rows.append([at,i,int(j),float(S[i,j])])
        df=pd.DataFrame(rows, columns=["actor_type","actor_idx","topic","score"])
        if at in agg and len(agg[at])>0:
            g=agg[at]
            df=df.merge(g, left_on=["actor_idx","topic"], right_on=[at,"topic"], how="left")
            if at in df.columns: df=df.drop(columns=[at])
        else:
            df["events"]=np.nan; df["support"]=np.nan
        df["direction"]=np.sign(df["support"].fillna(0.0))
        df["power"]=df["score"].abs()*df["support"].abs().fillna(0.0)
        outs.append(df)
    if len(outs)>0:
        res=pd.concat(outs, ignore_index=True)
        res.to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_actor_topic.parquet"), index=False)
    else:
        pd.DataFrame(columns=["actor_type","actor_idx","topic","score","events","support","direction","power"]).to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_actor_topic.parquet"), index=False)
    dec_rows=[]
    te=F.normalize(model.infl.topic_proj(model.infl.topic_table.weight).detach().cpu(),dim=-1)
    for at in ["legislator_term","committee","donor","lobby_firm"]:
        if at not in emb:
            continue
        A=F.normalize(emb[at],dim=-1)
        for i in range(min(A.size(0), 512)):
            best=torch.topk((A[i].unsqueeze(0)@te.T).squeeze(0), k=min(3,te.size(0))).indices.tolist()
            for tj in best:
                topic_vec=te[tj]
                dec=decompose_actor_topic(model.backbone, model.backbone.rel_attn, per_rel, at, i, topic_vec)
                for rel, val in dec.items():
                    dec_rows.append([at,i,int(tj),rel,float(val)])
    if len(dec_rows)>0:
        pd.DataFrame(dec_rows, columns=["actor_type","actor_idx","topic","relation","contribution"]).to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_actor_topic_decomp.parquet"), index=False)
    else:
        pd.DataFrame(columns=["actor_type","actor_idx","topic","relation","contribution"]).to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_actor_topic_decomp.parquet"), index=False)

def build_overall_influence_series(data, emb, save_prefix):
    outs=[]
    for at in ["legislator_term","committee","donor","lobby_firm"]:
        if at in emb:
            A=emb[at]
            infl=torch.norm(A, dim=-1)
            df=pd.DataFrame({"actor_type":at,"actor_idx":np.arange(A.size(0)),"influence":infl.detach().cpu().numpy()})
            outs.append(df)
    if len(outs)>0:
        pd.concat(outs, ignore_index=True).to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_overall_influence.parquet"), index=False)
    else:
        pd.DataFrame(columns=["actor_type","actor_idx","influence"]).to_parquet(os.path.join(CFG.save_dir, f"{save_prefix}_overall_influence.parquet"), index=False)

if __name__=="__main__":
    outs=run_train("data4.pt", save_prefix="leginflu_v3")
    print(json.dumps(outs, indent=2))
