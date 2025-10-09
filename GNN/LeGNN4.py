import os, math, json, random, torch, numpy as np, gc
from collections import defaultdict, Counter
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes, RemoveDuplicatedEdges
from tqdm import tqdm

# — device, seeds
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# — hparams
hidden_dim=124; drop_p=0.15
epochs_stage1=20; epochs_stage2=5
w_bill=1.0; w_stance=0.35; w_contrast=0.05; w_inf=0.5; w_dose=0.3; w_route=0.2
lambda_l2=1e-4; tau_contrast=0.7
max_pos_pairs_per_batch=64; neutral_cls=1; min_conf=3.0
ema_beta=0.95; bins_dose=8; cal_bins=15
accum_bill=8; accum_contrast=2; accum_stance=4

manual_time_cols_nodes={k:[-1] for k in ['bill','bill_version','legislator_term']}
manual_time_cols_edges={k:[-1] for k in [
    ('legislator_term','voted_on','bill_version'),
    ('committee','rev_read','bill_version'),
    ('donor','donated_to','legislator_term'),
    ('lobby_firm','lobbied','committee'),
    ('lobby_firm','lobbied','legislator_term'),
]}

# — data
data=torch.load('data4.pt', weights_only=False)
data=ToUndirected()(data); data=RemoveIsolatedNodes()(data); data=RemoveDuplicatedEdges()(data)
for store in data.node_stores+data.edge_stores:
    for k,v in list(store.items()):
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point: torch.nan_to_num_(v, nan=0.0, posinf=0.0, neginf=0.0)

def _l2_mean(t): return t.pow(2).mean()

def infer_time_cols(t,max_cols=8):
    if t is None or t.numel()==0: return []
    x=t.view(-1,1).float() if t.dim()==1 else t.float(); n,d=x.size()
    var=x.var(0); non=(var>0);
    if non.sum()==0: return []
    xnc=x[:,non]; j=torch.arange(d, device=x.device)[non]
    mn=xnc.quantile(0.01,0); mx=xnc.quantile(0.99,0)
    sec=(mn>60*60*24*365*50)&(mx<60*60*24*365*300); nano=(mn>1e17)&(mx<1e20)
    cand=j[(sec|nano)].tolist()
    if not cand:
        near=(mn>1900)&(mx<2300); cand=j[near].tolist()
    if not cand:
        m=xnc.mean(0); s=(xnc-m).pow(4).mean(0)/(xnc.var(0)+1e-6).pow(2)
        cand=j[torch.argsort(s,descending=True)][:max_cols].tolist()
    return cand[:max_cols]

def get_time_cols_for_node(nt):
    if nt in manual_time_cols_nodes: return manual_time_cols_nodes[nt]
    return infer_time_cols(data[nt].get('x',None))

def get_time_cols_for_edge(et):
    if et in manual_time_cols_edges: return manual_time_cols_edges[et]
    return infer_time_cols(data[et].get('edge_attr',None))

def pick_ts_from_cols(mat, cols):
    if mat is None or not cols: return None
    m=mat.view(-1,1) if mat.dim()==1 else mat
    cols=[c for c in cols if c<m.size(1)]
    if not cols: return None
    v=torch.nan_to_num(m[:,cols], nan=0.0, posinf=0.0, neginf=0.0).float()
    for c in range(v.size(1)):
        col=v[:,c];
        if col.max()>1e17: v[:,c]=col/1e9
    nz=(v>0).float()
    mins=torch.where(nz.bool(), v, torch.full_like(v,1e30)).min(1).values
    mins[mins==1e30]=0.0
    return mins

def sigmoid_decay(ts_now, ts_events, half_life_days=365.0):
    ts_events=torch.as_tensor(ts_events, dtype=torch.float32)
    if ts_events.dim()==0: ts_events=ts_events.unsqueeze(0)
    ts_now=torch.as_tensor(ts_now, dtype=torch.float32, device=ts_events.device)
    dd=(ts_now-ts_events)/86400.0; dd=torch.clamp(dd,min=0.0)
    k=math.log(2.0)/max(float(half_life_days),1e-6)
    dec=torch.exp(-k*dd); dec=torch.nan_to_num(dec, nan=1.0, posinf=0.0, neginf=1.0)
    return dec

def cap_amount(a,cap):
    if a is None: return None
    return torch.clamp(a,max=cap)

def make_rng(dev,seed=42):
    g=torch.Generator(device=dev) if dev!='cpu' else torch.Generator()
    g.manual_seed(seed); return g

rng=make_rng('cpu'); rngg=make_rng(device)

# — labels/topics
y_raw=data['bill'].y.clone() if hasattr(data['bill'],'y') else torch.full((data['bill'].num_nodes,),-1,dtype=torch.long)
data['bill'].y=torch.where(y_raw>=0,y_raw,torch.full_like(y_raw,-1))
if hasattr(data['bill'],'cluster'):
    vc=torch.unique(data['bill'].cluster[data['bill'].cluster>=0]); K_topics=int(vc.max().item()+1) if vc.numel() else 0
else: K_topics=0

def _infer_success_labels_from_y(y):
    if y.numel()==0: return y
    out=torch.zeros_like(y); pos=(y==1)|(y==2)|(y==3); out[pos]=1; return out

y_success=data['bill'].y_success.clone() if hasattr(data['bill'],'y_success') else _infer_success_labels_from_y(data['bill'].y.clone())
data['bill'].y_success=y_success

# — bill ts
bill_ts=pick_ts_from_cols(data['bill'].get('x',None), get_time_cols_for_node('bill'))
bv_to_bill=None
if ('bill_version','is_version','bill') in data.edge_types:
    ei=data[('bill_version','is_version','bill')].edge_index
    bv_to_bill=torch.full((data['bill_version'].num_nodes,), -1, dtype=torch.long); bv_to_bill[ei[0]]=ei[1]

def _bill_ts_from_edges():
    if bv_to_bill is None: return None
    if ('bill_version','read','committee') in data.edge_types:
        e=data[('bill_version','read','committee')]
        ts=pick_ts_from_cols(e.get('edge_attr',None), get_time_cols_for_edge(('bill_version','read','committee')))
        if ts is not None:
            bvid=e.edge_index[0]; bid=bv_to_bill[bvid]; m=(bid>=0)
            if m.any():
                agg=torch.zeros(data['bill'].num_nodes); cnt=torch.zeros_like(agg)
                agg.index_add_(0, bid[m].cpu(), ts[m].cpu()); cnt.index_add_(0, bid[m].cpu(), torch.ones_like(ts[m]).cpu())
                mean=torch.where(cnt>0, agg/cnt, torch.zeros_like(agg));
                if mean.max()>0: return mean
    if ('legislator_term','voted_on','bill_version') in data.edge_types:
        e=data[('legislator_term','voted_on','bill_version')]
        ts=pick_ts_from_cols(e.get('edge_attr',None), get_time_cols_for_edge(('legislator_term','voted_on','bill_version')))
        if ts is not None:
            bvid=e.edge_index[1]; bid=bv_to_bill[bvid]; m=(bid>=0)
            if m.any():
                agg=torch.zeros(data['bill'].num_nodes); cnt=torch.zeros_like(agg)
                agg.index_add_(0, bid[m].cpu(), ts[m].cpu()); cnt.index_add_(0, bid[m].cpu(), torch.ones_like(ts[m]).cpu())
                mean=torch.where(cnt>0, agg/cnt, torch.zeros_like(agg))
                if mean.max()>0: return mean
    return None

if bill_ts is None or bill_ts.max()==0:
    tmp=_bill_ts_from_edges(); bill_ts=tmp if tmp is not None else torch.zeros(data['bill'].num_nodes)
data['bill'].ts=bill_ts.float()

# — temporal split
def temporal_holdout_by_year(val_year=None):
    ts=data['bill'].ts
    if ts is None or ts.max()==0:
        n=data['bill'].num_nodes; k=max(1,int(0.15*n))
        perm=torch.randperm(n, generator=rng); val=perm[:k]
        tm=torch.ones(n,dtype=torch.bool); tm[val]=False; vm=~tm
        return tm&(data['bill'].y_success>=0), vm&(data['bill'].y_success>=0)
    year=torch.clamp((1970+ts/31557600.0).long(),1970,2100)
    vy=torch.quantile(year.float(),0.85).long().item() if val_year is None else val_year
    tm=(year<=vy)&(data['bill'].y_success>=0); vm=(year==vy+1)&(data['bill'].y_success>=0)
    if vm.sum()<max(64,int(0.02*year.numel())):
        perm=torch.randperm(year.numel(), generator=rng); val=perm[:max(64,int(0.15*year.numel()))]
        tm=torch.ones_like(tm); tm[val]=False; vm=~tm; tm=tm&(data['bill'].y_success>=0); vm=vm&(data['bill'].y_success>=0)
    return tm, vm

train_mask, val_mask=temporal_holdout_by_year(None)

# — encoders
encoders=nn.ModuleDict(); proj=nn.ModuleDict(); proj_dropout=nn.ModuleDict()
for nt in data.node_types:
    if 'x' in data[nt]:
        in_dim=data[nt].x.size(-1); encoders[nt]=nn.Identity(); proj[nt]=nn.Linear(in_dim, hidden_dim, bias=False); proj_dropout[nt]=nn.Dropout(drop_p)
    else:
        num=data[nt].num_nodes; encoders[nt]=nn.Embedding(num, hidden_dim); proj[nt]=nn.Identity(); proj_dropout[nt]=nn.Dropout(drop_p)
norms=nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in data.node_types})

class EdgeTimeEncoder(nn.Module):
    def __init__(self,in_dim):
        super().__init__(); h=max(8,min(64,in_dim)); self.mlp=nn.Identity() if in_dim<=0 else nn.Sequential(nn.Linear(in_dim,h), nn.GELU(), nn.Linear(h,h)); self.out_dim=0 if in_dim<=0 else h
    def forward(self,eattr): return None if eattr is None else self.mlp(eattr)

edge_attr_dims={et:(data[et].edge_attr.size(-1) if 'edge_attr' in data[et] else 0) for et in data.edge_types}
edge_encoders=nn.ModuleDict({str(et):EdgeTimeEncoder(edge_attr_dims[et]) for et in data.edge_types})

class EdgeGatedSAGEConv(MessagePassing):
    def __init__(self, in_src, in_dst, out_ch, e_dim=0, aggr='mean'):
        super().__init__(aggr=aggr)
        self.lin_src = nn.Linear(in_src, out_ch, bias=False)
        self.lin_dst = nn.Linear(in_dst, out_ch, bias=True)
        self.e_dim = int(e_dim)  # <- store for runtime padding
        gdim = in_src + in_dst + (self.e_dim if self.e_dim > 0 else 0)
        self.gate = nn.Linear(gdim, 1)
        self.lin_upd = nn.Linear(out_ch, out_ch)
        self.last_alpha = None

    def forward(self, x, edge_index, edge_attr=None, time_decay=None):
        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        xs = self.lin_src(x_src)
        xd = self.lin_dst(x_dst)

        # ensure edge_attr has the expected width if this conv was built with e_dim>0
        if (edge_attr is None or edge_attr.numel() == 0) and self.e_dim > 0:
            edge_attr = xs.new_zeros(edge_index.size(1), self.e_dim)

        a_in = [x_src[edge_index[0]], x_dst[edge_index[1]]]
        if edge_attr is not None:
            a_in.append(edge_attr)

        alpha = torch.sigmoid(self.gate(torch.cat(a_in, dim=-1))).view(-1)
        if time_decay is not None:
            alpha = alpha * time_decay.view(-1)

        self.last_alpha = alpha.detach()
        out = self.propagate(edge_index, x=xs, alpha=alpha, size=(x_src.size(0), x_dst.size(0)))
        out = out + xd
        return self.lin_upd(F.gelu(out))

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)

def build_temporal_decay_for_batch(batch,et):
    e=batch[et]; n_e=e.edge_index.size(1); eattr=e.get('edge_attr',None)
    cols=get_time_cols_for_edge(et); ts=pick_ts_from_cols(eattr, cols) if eattr is not None else None
    if ts is None or ts.numel()!=n_e: return torch.ones(n_e, device=device)
    now=float(data['bill'].ts.max().item() if 'ts' in data['bill'] else 0.0)
    dec=sigmoid_decay(now, ts);
    if dec.numel()!=n_e or not torch.isfinite(dec).all(): return torch.ones(n_e, device=device)
    return dec.to(device)

conv1=HeteroConv({et: EdgeGatedSAGEConv(hidden_dim,hidden_dim,hidden_dim, edge_encoders[str(et)].out_dim) for et in data.edge_types}, aggr='sum')
conv2=HeteroConv({et: EdgeGatedSAGEConv(hidden_dim,hidden_dim,hidden_dim, edge_encoders[str(et)].out_dim) for et in data.edge_types}, aggr='sum')
post_conv_dropout1=nn.ModuleDict({nt: nn.Dropout(drop_p) for nt in data.node_types})
post_conv_dropout2=nn.ModuleDict({nt: nn.Dropout(drop_p) for nt in data.node_types})

topic_emb=nn.Embedding(max(K_topics,1), hidden_dim) if K_topics>0 else None
bill_head_bin=nn.Linear(hidden_dim,2)
stance_head_LT=nn.Linear(hidden_dim, K_topics*3).to(device) if K_topics>0 else None

for md in (encoders,proj,proj_dropout,norms,post_conv_dropout1,post_conv_dropout2,edge_encoders,conv1,conv2,bill_head_bin):
    md.to(device)
if topic_emb is not None: topic_emb.to(device)
if stance_head_LT is not None: stance_head_LT.to(device)

# — neighbors
num_neighbors={}
for et in data.edge_types:
    if et in [('legislator_term','voted_on','bill_version'), ('bill_version','read','committee'), ('committee','rev_read','bill_version')]:
        num_neighbors[et]=[6,4]
    elif et==('bill_version','priorVersion','bill_version'):
        num_neighbors[et]=[12,8]
    elif et[0] in ['donor','lobby_firm']:
        num_neighbors[et]=[6,3]
    else:
        num_neighbors[et]=[12,6]

train_bill_ids=torch.arange(data['bill'].num_nodes)[train_mask]
val_bill_ids=torch.arange(data['bill'].num_nodes)[val_mask]

bill_loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('bill',train_bill_ids), batch_size=1024, shuffle=True)
val_bill_loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('bill',val_bill_ids), batch_size=2048, shuffle=False)

num_neighbors_bv={('bill_version','priorVersion','bill_version'):[4,2], ('bill_version','is_version','bill'):[1,0]}
for et in data.edge_types: num_neighbors_bv.setdefault(et,[0,0])

participating_bv=torch.tensor([],dtype=torch.long)
if ('bill_version','priorVersion','bill_version') in data.edge_types:
    ei=data[('bill_version','priorVersion','bill_version')].edge_index
    participating_bv=torch.unique(torch.cat([ei[0],ei[1]]))

bv_loader = None  # replaced by cached contrast

if ('bill_version','is_version','bill') in data.edge_types:
    ei=data[('bill_version','is_version','bill')].edge_index
    bv_to_bill=torch.full((data['bill_version'].num_nodes,), -1, dtype=torch.long); bv_to_bill[ei[0]]=ei[1]

def final_passage_mask_global():
    if ('legislator_term','voted_on','bill_version') not in data.edge_types: return None
    if ('bill_version','priorVersion','bill_version') not in data.edge_types: return None
    ve=data[('legislator_term','voted_on','bill_version')].edge_index
    pv=data[('bill_version','priorVersion','bill_version')].edge_index
    has_child=torch.zeros(data['bill_version'].num_nodes,dtype=torch.bool); has_child[pv[0]]=True
    terminal=~has_child; return terminal[ve[1]]

def _extract_vote_sign(eattr):
    if eattr is None: return None
    aye_ix,no_ix=None,None
    for j in range(eattr.size(1)):
        col=eattr[:,j]
        if torch.all((col==0)|(col==1)):
            if aye_ix is None: aye_ix=j
            elif no_ix is None: no_ix=j
    if aye_ix is not None and no_ix is not None: return eattr[:,aye_ix].float()-eattr[:,no_ix].float()
    if aye_ix is not None: return 2.0*eattr[:,aye_ix].float()-1.0
    return None

def build_lt_topic_labels_fold(train_mask_bills):
    if K_topics==0 or ('legislator_term','voted_on','bill_version') not in data.edge_types or bv_to_bill is None or not hasattr(data['bill'],'cluster'):
        return None,None
    ei=data[('legislator_term','voted_on','bill_version')].edge_index
    eattr=data[('legislator_term','voted_on','bill_version')].get('edge_attr',None)
    v=_extract_vote_sign(eattr); v=torch.zeros(ei.size(1)) if v is None else v.to(torch.float32)
    tcols=get_time_cols_for_edge(('legislator_term','voted_on','bill_version')); vts=pick_ts_from_cols(eattr,tcols) if tcols else None
    vts=torch.zeros(ei.size(1)) if vts is None else vts
    bvid=ei[1]; bid=bv_to_bill[bvid]
    keep_bill=(bid>=0)&train_mask_bills[bid]
    keep_time=torch.ones_like(keep_bill,dtype=torch.bool)
    if 'ts' in data['bill'] and data['bill'].ts.numel()==data['bill'].num_nodes:
        keep_time=(vts>0)&(data['bill'].ts[bid]>0)&(vts<=data['bill'].ts[bid])
    keep_final=final_passage_mask_global()
    keep=keep_bill & keep_time if keep_final is None else keep_bill & keep_time & keep_final
    lt=ei[0][keep]; top=data['bill'].cluster[bid[keep]].long(); val=v[keep]
    lt_n=data['legislator_term'].num_nodes
    labs=torch.full((lt_n,K_topics),-100,dtype=torch.long); conf=torch.zeros((lt_n,K_topics))
    idx=lt*K_topics+top; sums=torch.zeros(lt_n*K_topics); cnt=torch.zeros(lt_n*K_topics)
    sums.index_add_(0, idx, val); cnt.index_add_(0, idx, torch.ones_like(val))
    sums=sums.view(lt_n,K_topics); cnt=cnt.view(lt_n,K_topics)
    thr=torch.clamp(cnt.sqrt(),min=1.0)
    lab=torch.where(sums>0.5*thr, torch.tensor(2), torch.where(sums<-0.5*thr, torch.tensor(0), torch.tensor(1)))
    m=cnt>=min_conf; labs=torch.where(m,lab,torch.full_like(lab,-100)); conf=cnt
    labeled=(labs>=0).any(1).sum().item()
    if labeled<max(256,int(0.01*lt_n)):
        lt=ei[0]; top=data['bill'].cluster[bv_to_bill[ei[1]].clamp_min(0)].long(); v_soft=v.to(torch.float32)
        idx=lt*K_topics+top; sums=torch.zeros(lt_n*K_topics); cnt=torch.zeros(lt_n*K_topics)
        sums.index_add_(0, idx, v_soft); cnt.index_add_(0, idx, torch.ones_like(v_soft))
        sums=sums.view(lt_n,K_topics); cnt=cnt.view(lt_n,K_topics)
        thr=torch.clamp(cnt.sqrt(),min=1.0)
        lab=torch.where(sums>0.5*thr, torch.tensor(2), torch.where(sums<-0.5*thr, torch.tensor(0), torch.tensor(1)))
        m=cnt>=max(1.0, min_conf*0.5); labs=torch.where(m,lab,torch.full_like(lab,-100)); conf=cnt
    return labs, conf

lt_topic_label, lt_topic_conf=build_lt_topic_labels_fold(train_mask)

def committee_swing_labels():
    if K_topics==0 or bv_to_bill is None or ('bill_version','read','committee') not in data.edge_types or ('legislator_term','voted_on','bill_version') not in data.edge_types or not hasattr(data['bill'],'cluster'):
        return None,None
    ce=data[('bill_version','read','committee')]; ve=data[('legislator_term','voted_on','bill_version')]
    c_ts=pick_ts_from_cols(ce.get('edge_attr',None), get_time_cols_for_edge(('bill_version','read','committee')))
    v_ts=pick_ts_from_cols(ve.get('edge_attr',None), get_time_cols_for_edge(('legislator_term','voted_on','bill_version')))
    if c_ts is None or v_ts is None: return None,None
    v_sign=_extract_vote_sign(ve.get('edge_attr',None))
    if v_sign is None: return None,None
    bvid_read=ce.edge_index[0]; bid_read=bv_to_bill[bvid_read]; cm_read=ce.edge_index[1]
    bvid_vote=ve.edge_index[1]; bid_vote=bv_to_bill[bvid_vote]
    per_bill_reads=defaultdict(list)
    for i in range(ce.edge_index.size(1)):
        b=int(bid_read[i].item());
        if b>=0: per_bill_reads[b].append((float(c_ts[i].item()), int(cm_read[i].item())))
    per_bill_votes=defaultdict(list)
    for i in range(ve.edge_index.size(1)):
        b=int(bid_vote[i].item());
        if b>=0: per_bill_votes[b].append((float(v_ts[i].item()), float(v_sign[i].item())))
    labs=torch.full((data['committee'].num_nodes,K_topics),-100,dtype=torch.long); conf=torch.zeros((data['committee'].num_nodes,K_topics))
    for b,seq in per_bill_reads.items():
        seq.sort(key=lambda x:x[0]); votes=per_bill_votes.get(b,[])
        if not votes: continue
        votes.sort(key=lambda x:x[0]); topic=int(data['bill'].cluster[b].item())
        vt=torch.tensor([t for (t,_) in votes]); vs=torch.tensor([s for (_,s) in votes])
        for t_read,cm in seq:
            m_before=vs[vt<t_read].sum().item() if (vt<t_read).any() else 0.0
            window=(vt>=t_read)&(vt<=t_read+60*86400.0); m_after=vs[window].sum().item() if window.any() else m_before
            swing=m_after-m_before; cls=2 if swing>0 else (0 if swing<0 else 1)
            if labs[cm,topic] < 0: labs[cm,topic]=cls; conf[cm,topic]=1
            else:
                prev=labs[cm,topic].item(); n=conf[cm,topic].item(); new=round((prev*n+cls)/(n+1))
                labs[cm,topic]=torch.tensor(new); conf[cm,topic]+=1
    return labs, conf

comm_topic_label, comm_topic_conf=committee_swing_labels()

def _aggregate_money_stance(actor_et, target_node_type, amount_col=0, time_cols=None):
    if K_topics==0 or actor_et not in data.edge_types: return None,None
    ei=data[actor_et].edge_index; eattr=data[actor_et].get('edge_attr',None)
    ts_cols=get_time_cols_for_edge(actor_et) if time_cols is None else time_cols
    ts=pick_ts_from_cols(eattr, ts_cols); now=torch.tensor(float(data['bill'].ts.max().item() if 'ts' in data['bill'] else 0.0))
    decay=sigmoid_decay(now, ts, 365.0); amt=eattr[:,amount_col] if (eattr is not None and eattr.size(1)>amount_col) else torch.ones(ei.size(1))
    if amt.numel()>0: amt=cap_amount(amt, cap=torch.quantile(amt,0.99))
    tgt=ei[1]
    if target_node_type=='legislator_term': labs,confs=lt_topic_label,lt_topic_conf
    elif target_node_type=='committee': labs,confs=comm_topic_label,comm_topic_conf
    else: return None,None
    if labs is None: return None,None
    labs_t=labs[tgt]; decay=torch.ones_like(amt) if decay is None else decay
    scores=defaultdict(lambda: torch.zeros(K_topics)); weights=defaultdict(lambda: torch.zeros(K_topics))
    for i in range(ei.size(1)):
        cls=labs_t[i]
        if cls.numel()==0 or (cls<0).all(): continue
        w=(amt[i]*decay[i]).item(); nid=ei[0,i].item()
        for t in range(K_topics):
            c=int(cls[t].item());
            if c<0: continue
            scores[nid][t]+= (2.0 if c==2 else 0.0 if c==0 else 1.0)*w; weights[nid][t]+=w
    N=data[actor_et[0]].num_nodes
    out_lab=torch.full((N,K_topics),-100,dtype=torch.long); out_conf=torch.zeros((N,K_topics))
    for nid in range(N):
        if nid not in scores: continue
        s=scores[nid]; w=weights[nid]; mean=torch.where(w>0, s/torch.clamp(w,min=1e-6), torch.full_like(s,float('nan')))
        cls=torch.where(mean>1.4,2, torch.where(mean<0.6,0,1)).long()
        out_lab[nid]=torch.where(w>0, cls, torch.full_like(cls,-100)); out_conf[nid]=w
    return out_lab, out_conf

don_topic_label, don_topic_conf=_aggregate_money_stance(('donor','donated_to','legislator_term'),'legislator_term', amount_col=0)
lob_lt_label,  lob_lt_conf =_aggregate_money_stance(('lobby_firm','lobbied','legislator_term'),'legislator_term', amount_col=0)
lob_cm_label,  lob_cm_conf =_aggregate_money_stance(('lobby_firm','lobbied','committee'),'committee', amount_col=0)

def labeled_indices_from(lab_mat, n):
    if lab_mat is None: return torch.tensor([],dtype=torch.long)
    ok=(lab_mat>=0).any(1); idx=torch.arange(n)[ok]
    return idx if idx.numel()>0 else torch.tensor([],dtype=torch.long)

eligible_cm=labeled_indices_from(comm_topic_label, data['committee'].num_nodes) if 'committee' in data.node_types else torch.tensor([],dtype=torch.long)
eligible_dn=labeled_indices_from(don_topic_label, data['donor'].num_nodes) if 'donor' in data.node_types else torch.tensor([],dtype=torch.long)
eligible_lb=labeled_indices_from(lob_lt_label if lob_lt_label is not None else lob_cm_label, data['lobby_firm'].num_nodes) if 'lobby_firm' in data.node_types else torch.tensor([],dtype=torch.long)
eligible_lt=labeled_indices_from(lt_topic_label, data['legislator_term'].num_nodes) if 'legislator_term' in data.node_types else torch.tensor([],dtype=torch.long)

lt_loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('legislator_term', eligible_lt), batch_size=248, shuffle=True) if eligible_lt.numel()>0 else None
comm_loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('committee', eligible_cm), batch_size=248, shuffle=True) if eligible_cm.numel()>0 else None
don_loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('donor', eligible_dn), batch_size=248, shuffle=True) if eligible_dn.numel()>0 else None
lob_loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('lobby_firm', eligible_lb), batch_size=248, shuffle=True) if eligible_lb.numel()>0 else None

# — positive pairs (global)
pos_pairs=[]
if ('bill_version','priorVersion','bill_version') in data.edge_types:
    ei=data[('bill_version','priorVersion','bill_version')].edge_index; a,b=ei[0].tolist(), ei[1].tolist()
    pos_pairs=list(set(tuple(sorted(p)) for p in zip(a,b)))
    if len(pos_pairs)>200_000:
        keep=torch.randperm(len(pos_pairs), generator=rng, device='cpu')[:200_000].tolist()
        pos_pairs=[pos_pairs[i] for i in keep]

stance_head_comm  = nn.Linear(hidden_dim, K_topics*3).to(device) if K_topics>0 and comm_topic_label is not None else None
stance_head_donor = nn.Linear(hidden_dim, K_topics*3).to(device) if K_topics>0 and don_topic_label  is not None else None
stance_head_lobby = nn.Linear(hidden_dim, K_topics*3).to(device) if K_topics>0 and (lob_lt_label is not None or lob_cm_label is not None) else None

def ece(probs, targets, n_bins=15):
    conf, pred=probs.max(1); acc=pred.eq(targets); bins=torch.linspace(0,1,n_bins+1)
    e=torch.zeros(1)
    for i in range(n_bins):
        m=(conf>bins[i])&(conf<=bins[i+1])
        if m.any(): e += (m.float().mean() * torch.abs(acc[m].float().mean()-conf[m].mean()))
    return e.item()

def brier(probs, targets):
    oh=F.one_hot(targets, num_classes=probs.size(1)).float()
    return ((probs-oh)**2).mean().item()

class TemperatureScaling(nn.Module):
    def __init__(self): super().__init__(); self.log_T=nn.Parameter(torch.zeros(()))
    def forward(self,logits): return logits/torch.exp(self.log_T)
    def fit(self,logits,targets,lr=5e-2,max_iter=256):
        optT=torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter)
        logits=logits.detach(); targets=targets.detach()
        def _c(): optT.zero_grad(); loss=F.cross_entropy(self.forward(logits),targets); loss.backward(); return loss
        optT.step(_c); return torch.exp(self.log_T).item()

def collect_bill_logits_embeddings(head, ids):
    loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('bill', ids), batch_size=4096, shuffle=False)
    all_logits,all_targets,all_topics=[],[],[]
    for batch in loader:
        batch=batch.to(device)
        _,h2,_,_=forward_block(batch); n_b=batch['bill'].batch_size
        tid=(batch['bill'].cluster[:n_b].to(device).long() if (K_topics>0 and hasattr(batch['bill'],'cluster')) else torch.zeros(n_b,dtype=torch.long,device=device))
        logits2=head(h2['bill'][:n_b], tid); targets=batch['bill'].y_success[:n_b].to(device)
        all_logits.append(logits2.detach().cpu()); all_targets.append(targets.detach().cpu()); all_topics.append(tid.detach().cpu())
    return torch.cat(all_logits,0), torch.cat(all_targets,0), torch.cat(all_topics,0)

class TopicConditionedBillHead(nn.Module):
    def __init__(self, base_head, topic_emb):
        super().__init__(); self.base_head=base_head; self.topic_emb=topic_emb; self.mix=nn.Linear(hidden_dim,hidden_dim,bias=False)
    def forward(self,h,topic_ids):
        if self.topic_emb is not None: h=h+self.mix(self.topic_emb(topic_ids))
        return self.base_head(h)

bill_head_tc=TopicConditionedBillHead(bill_head_bin.to(device), topic_emb if topic_emb is not None else nn.Embedding(1,hidden_dim).to(device)).to(device)

class CalibratedPerTopic(nn.Module):
    def __init__(self, head, K):
        super().__init__(); self.head=head; self.scalers=nn.ModuleList([TemperatureScaling() for _ in range(max(K,1))]); self.K=max(K,1)
    def forward(self,h,topic_ids):
        logits=self.head(h,topic_ids)
        if self.K==1: return self.scalers[0](logits)
        out=torch.empty_like(logits)
        for t in range(self.K):
            m=(topic_ids==t)
            if m.any(): out[m]=self.scalers[t](logits[m])
        if (~torch.isfinite(out)).any(): out=logits
        return out
    def fit(self,loader_fn):
        logits,targets,topics=loader_fn()
        for t in range(self.K):
            m=(topics==t)
            lt,yt=logits[m],targets[m]
            if lt.size(0)>=50:
                self.scalers[t].to('cpu'); _=self.scalers[t].fit(lt.to('cpu'), yt.to('cpu'))
            else:
                with torch.no_grad(): self.scalers[t].log_T.zero_()

class DoseResponseHead(nn.Module):
    def __init__(self, hidden_dim, K, bins=8):
        super().__init__(); self.topic_emb=nn.Embedding(max(K,1),hidden_dim)
        self.f=nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
        self.g=nn.Sequential(nn.Linear(hidden_dim+bins,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1)); self.bins=bins
    def forward(self,h_bill,t_bins,topic_ids):
        te=self.topic_emb(topic_ids); x=self.f(h_bill+te); x=torch.cat([x,t_bins],-1); return self.g(x).squeeze(-1)

class RouteEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__(); self.gru=nn.GRU(hidden_dim,hidden_dim,batch_first=True); self.head=nn.Linear(hidden_dim,1)
    def forward(self,seq_emb,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(seq_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out,_=self.gru(packed); out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
        last=out[torch.arange(out.size(0)), lengths-1]; return last, self.head(last).squeeze(-1)

def monotonic_penalty(yb): dif=yb[:,1:]-yb[:,:-1]; return torch.relu(-dif).mean()

route_enc=RouteEncoder(hidden_dim).to(device)
dose_head=DoseResponseHead(hidden_dim, K_topics, bins=bins_dose).to(device)

proj_contrast=nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,64)).to(device)

params=list(encoders.parameters())+list(proj.parameters())+list(conv1.parameters())+list(conv2.parameters())+list(norms.parameters())+list(post_conv_dropout1.parameters())+list(post_conv_dropout2.parameters())+list(proj_dropout.parameters())+list(edge_encoders.parameters())+list(topic_emb.parameters() if topic_emb is not None else [])+list(bill_head_tc.parameters())+list(dose_head.parameters())+list(route_enc.parameters())+list(stance_head_LT.parameters() if stance_head_LT is not None else [])+list(stance_head_comm.parameters() if stance_head_comm is not None else [])+list(stance_head_donor.parameters() if stance_head_donor is not None else [])+list(stance_head_lobby.parameters() if stance_head_lobby is not None else [])+list(proj_contrast.parameters())
opt=torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4, betas=(0.9,0.999))
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage1+epochs_stage2, eta_min=3e-4)

def edge_modules_inputs(batch):
    eattr_dict = {}
    tdec_dict = {}
    for et in batch.edge_types:
        e = batch[et]
        # fast path for empty edge sets in the mini-batch
        if e.edge_index.numel() == 0:
            eattr_dict[et] = None
            tdec_dict[et] = None
            continue

        raw_eattr = e.get('edge_attr', None)
        enc = edge_encoders[str(et)]
        # encode if present
        eenc = enc(raw_eattr.to(device) if raw_eattr is not None else None)

        if (eenc is None or eenc.numel() == 0) and enc.out_dim > 0:
            eenc = torch.zeros(e.edge_index.size(1), enc.out_dim, device=device)

        eattr_dict[et] = eenc

        tdec = build_temporal_decay_for_batch(batch, et)
        tdec_dict[et] = tdec.to(device) if tdec is not None else None
    return eattr_dict, tdec_dict


def forward_block(batch, edge_w_override=None):
    edge_index_dict={k:v.to(device) for k,v in batch.edge_index_dict.items()}
    hb={}
    for nt in batch.node_types:
        if 'x' in batch[nt]:
            hb[nt]=proj_dropout[nt](proj[nt](encoders[nt](batch[nt].x.to(device))))
        else:
            idx_map=batch[nt].n_id if hasattr(batch[nt],'n_id') else torch.arange(batch[nt].num_nodes, device=device)
            hb[nt]=proj_dropout[nt](encoders[nt](idx_map.to(device)))
    eattr_dict,tdec_dict=edge_modules_inputs(batch)
    out1={}
    for et,conv in conv1.convs.items():
        xs=hb[et[0]]; xd=hb[et[2]]
        ew=edge_w_override.get(et).to(device) if (edge_w_override is not None and et in edge_w_override and edge_w_override[et] is not None) else None
        ea=eattr_dict[et]; td=tdec_dict[et]
        if td is None: td=torch.ones(edge_index_dict[et].size(1), device=device)
        if ew is not None: td=td*ew
        out1[et]=conv((xs,xd), edge_index_dict[et], edge_attr=ea, time_decay=td)
    h1={}; agg1=defaultdict(list)
    for et,t in out1.items(): agg1[et[2]].append(t)
    for nt in hb:
        if nt in agg1: h=torch.stack(agg1[nt],0).sum(0); h1[nt]=norms[nt](F.gelu(hb[nt]+post_conv_dropout1[nt](h)))
        else: h1[nt]=norms[nt](F.gelu(hb[nt]))
    out2={}; alphas={}
    for et,conv in conv2.convs.items():
        xs=h1[et[0]]; xd=h1[et[2]]
        ew=edge_w_override.get(et).to(device) if (edge_w_override is not None and et in edge_w_override and edge_w_override[et] is not None) else None
        ea=eattr_dict[et]; td=tdec_dict[et]
        if td is None: td=torch.ones(edge_index_dict[et].size(1), device=device)
        if ew is not None: td=td*ew
        out2[et]=conv((xs,xd), edge_index_dict[et], edge_attr=ea, time_decay=td); alphas[et]=conv.last_alpha
    h2={}; agg2=defaultdict(list)
    for et,t in out2.items(): agg2[et[2]].append(t)
    for nt in h1:
        if nt in agg2: h=torch.stack(agg2[nt],0).sum(0); h2[nt]=norms[nt](F.gelu(h1[nt]+post_conv_dropout2[nt](h)))
        else: h2[nt]=norms[nt](F.gelu(h1[nt]))
    return hb,h2,alphas,edge_index_dict

def build_treatment_bins_for_batch(batch,h2_bill,bins=8):
    if 'bill' not in batch.node_types:
        return torch.zeros(0,bins,device=device), torch.zeros(0,dtype=torch.long,device=device), torch.linspace(0,1,steps=bins+1,device=device)
    n_b=batch['bill'].batch_size; score=torch.zeros(n_b, device=device)
    if ('legislator_term','voted_on','bill_version') in batch.edge_types and ('bill_version','is_version','bill') in batch.edge_types:
        v_ei=batch[('legislator_term','voted_on','bill_version')].edge_index
        isv=batch[('bill_version','is_version','bill')].edge_index
        bv2b=torch.full((batch['bill_version'].num_nodes,), -1, device=device, dtype=torch.long); bv2b[isv[0]]=isv[1]
        b_of_vote=bv2b[v_ei[1]]; keep=(b_of_vote>=0)&(b_of_vote<n_b); bkeep=b_of_vote[keep]
        def sum_amount_decay(et):
            if et not in batch.edge_types: return None
            e=batch[et]; eattr=e.get('edge_attr',None)
            if eattr is None: return None
            ts=pick_ts_from_cols(eattr, get_time_cols_for_edge(et)); ts=ts.to(device) if ts is not None else torch.zeros(e.edge_index.size(1), device=device)
            now=torch.tensor(float(data['bill'].ts.max().item() if 'ts' in data['bill'] else 0.0), device=device)
            decay=sigmoid_decay(now, ts, 365.0); decay=torch.ones_like(ts) if decay is None else decay
            amt=eattr[:,0].to(device) if eattr.size(1)>0 else torch.ones_like(ts)
            amt=torch.clamp(amt, max=torch.quantile(amt,0.99))
            return e.edge_index, amt*decay
        d_lt=sum_amount_decay(('donor','donated_to','legislator_term'))
        l_lt=sum_amount_decay(('lobby_firm','lobbied','legislator_term'))
        l_cm=sum_amount_decay(('lobby_firm','lobbied','committee'))
        if d_lt is not None:
            eidx,val=d_lt; lt_votes=v_ei[0][keep]; lt_in_batch=lt_votes.unique()
            mask=torch.isin(eidx[1].to(device), lt_in_batch)
            cnt=torch.zeros(batch['legislator_term'].num_nodes, device=device); cnt.index_add_(0, eidx[1][mask], val[mask])
            score.index_add_(0, bkeep, cnt[lt_votes])
        if l_lt is not None:
            eidx,val=l_lt; lt_votes=v_ei[0][keep]; lt_in_batch=lt_votes.unique()
            mask=torch.isin(eidx[1].to(device), lt_in_batch)
            cnt=torch.zeros(batch['legislator_term'].num_nodes, device=device); cnt.index_add_(0, eidx[1][mask], val[mask])
            score.index_add_(0, bkeep, cnt[lt_votes])
        if ('bill_version','read','committee') in batch.edge_types and l_cm is not None:
            eidx_cm,val_cm=l_cm
            isv=batch[('bill_version','is_version','bill')].edge_index
            bv2b=torch.full((batch['bill_version'].num_nodes,), -1, device=device, dtype=torch.long); bv2b[isv[0]]=isv[1]
            b_cm=bv2b[eidx_cm[0]]; maskb=(b_cm>=0)&(b_cm<n_b)
            cm_cnt=torch.zeros(batch['committee'].num_nodes, device=device); cm_cnt.index_add_(0, eidx_cm[1], val_cm)
            score.index_add_(0, b_cm[maskb], cm_cnt[eidx_cm[1][maskb]])
    pos=score[score>0]
    q=torch.quantile(pos, torch.linspace(0,1,steps=bins+1, device=device)) if pos.numel()>0 else torch.linspace(0,1,steps=bins+1, device=device)
    digit=torch.bucketize(score, q[1:-1])
    t_bins=F.one_hot(torch.clamp(digit,0,bins-1), num_classes=bins).float()
    topic_ids=batch['bill'].cluster[:n_b].long() if K_topics>0 and hasattr(batch['bill'],'cluster') else torch.zeros(n_b,dtype=torch.long,device=device)
    return t_bins, topic_ids, q.detach()

class BillPredictor(nn.Module):
    def __init__(self,head): super().__init__(); self.head=head
    def forward(self,h,topic_ids): return self.head(h,topic_ids)

bill_pred=BillPredictor(bill_head_tc).to(device)

def bill_loss(batch,h2):
    n_b=batch['bill'].batch_size
    tid=batch['bill'].cluster[:n_b].to(device).long() if (K_topics>0 and hasattr(batch['bill'],'cluster')) else torch.zeros(n_b,dtype=torch.long,device=device)
    logits=bill_pred(h2['bill'][:n_b], tid); y=batch['bill'].y_success[:n_b].to(device)
    return F.cross_entropy(logits,y,label_smoothing=0.02)+lambda_l2*_l2_mean(h2['bill'][:n_b])

def stance_training_step(batch,ntype,head,labs,conf):
    if head is None or labs is None or ntype not in batch.node_types: return None
    nid_local=batch[ntype].n_id[:batch[ntype].batch_size] if hasattr(batch[ntype],'n_id') else torch.arange(batch[ntype].batch_size, device=device)
    _,h2,_,_=forward_block(batch)
    if ntype not in h2 or h2[ntype].size(0)==0 or nid_local.numel()==0: return None
    out=head(h2[ntype][:nid_local.size(0)]).view(-1,K_topics,3)
    l=labs[nid_local.detach().cpu()]; c=conf[nid_local.detach().cpu()]
    m=l>=0
    if not m.any(): return None
    logits=out[m].to(device); targ=l[m].to(device); w=torch.clamp(c[m],max=10.0).to(device)
    ce=F.cross_entropy(logits,targ,reduction='none',label_smoothing=0.05)
    ls=(ce*w).mean(); p=F.softmax(out,dim=-1); nm=(l==neutral_cls)
    ent=(p[nm].max(-1).values-(1.0/3.0)).mean() if nm.any() else logits.new_tensor(0.0)
    return w_stance*ls+0.02*ent

def actor_targeted_masks(batch,actor_type,max_actors=256):
    masks={et: torch.ones(batch[et].edge_index.size(1), device=device) for et in batch.edge_types}
    if actor_type not in batch.node_types: return masks, torch.tensor([],dtype=torch.long,device=device)
    n=batch[actor_type].batch_size
    if n==0: return masks, torch.tensor([],dtype=torch.long,device=device)
    actors=torch.randperm(n, generator=rngg, device=device)[:min(max_actors,n)]
    incident=[]
    for et in batch.edge_types:
        if et[0]==actor_type or et[2]==actor_type: incident.append(et)
    for et in incident:
        ei=batch[et].edge_index
        m=torch.isin(ei[0],actors) if et[0]==actor_type else torch.isin(ei[1],actors)
        masks[et][m]=0.0
    actor_global=(batch[actor_type].n_id[:n])[actors] if hasattr(batch[actor_type],'n_id') else actors
    return masks, actor_global

influence_ema={
    'committee': torch.zeros(data['committee'].num_nodes, K_topics) if 'committee' in data.node_types and K_topics>0 else None,
    'donor': torch.zeros(data['donor'].num_nodes, K_topics) if 'donor' in data.node_types and K_topics>0 else None,
    'lobby_firm': torch.zeros(data['lobby_firm'].num_nodes, K_topics) if 'lobby_firm' in data.node_types and K_topics>0 else None,
    'legislator_term': torch.zeros(data['legislator_term'].num_nodes, K_topics) if 'legislator_term' in data.node_types and K_topics>0 else None,
}
alpha_snapshots=defaultdict(list)

def update_influence_topic(batch,actor_type,actor_ids_global,s_full,s_mask,topic_ids):
    if K_topics==0 or actor_type not in influence_ema: return
    buf=influence_ema[actor_type]
    if buf is None or actor_ids_global.numel()==0: return
    delta=(s_full-s_mask)
    if topic_ids.numel()!=delta.numel(): return
    for t in range(K_topics):
        m=(topic_ids==t)
        if not m.any(): continue
        d=delta[m].mean().item()
        for gid in actor_ids_global.tolist():
            if gid<buf.size(0): buf[gid,t]=ema_beta*buf[gid,t]+(1-ema_beta)*d

class MetricsLog:
    def __init__(self): self.hist={}
    def log(self,k,v): self.hist.setdefault(k,[]).append(v)
metrics=MetricsLog()

# — cached bill_version embeddings for contrast
@torch.no_grad()
def compute_bv_embeddings(sample_frac=0.5, max_nodes=120_000):
    if ('bill_version','priorVersion','bill_version') not in data.edge_types: return None,None
    nodes=participating_bv
    if sample_frac<1.0:
        k=max(1,int(sample_frac*nodes.numel()))
        idx=torch.randperm(nodes.numel(), generator=rng, device=nodes.device)[:k]
        nodes=nodes[idx]
    if max_nodes is not None and nodes.numel()>max_nodes:
        idx=torch.randperm(nodes.numel(), generator=rng, device=nodes.device)[:max_nodes]
        nodes=nodes[idx]
    nn_bv_embed={('bill_version','priorVersion','bill_version'):[2,0], ('bill_version','is_version','bill'):[1,0]}
    for et in data.edge_types: nn_bv_embed.setdefault(et,[0,0])
    loader=NeighborLoader(data, num_neighbors=nn_bv_embed, input_nodes=('bill_version', nodes), batch_size=2048, shuffle=False)
    Z=torch.empty(nodes.numel(), hidden_dim, device='cpu'); nid_map=torch.empty(nodes.numel(), dtype=torch.long, device='cpu'); first=True
    for batch in loader:
        batch=batch.to(device); _,h2,_,_=forward_block(batch); n=batch['bill_version'].batch_size
        g=batch['bill_version'].n_id[:n].to('cpu')
        if first:
            rev=-torch.ones(data['bill_version'].num_nodes, dtype=torch.long); rev[nodes.cpu()]=torch.arange(nodes.numel())
            first=False
        rows=rev[g]
        Z[rows]=h2['bill_version'][:n].detach().to('cpu')
        nid_map[rows]=g
    return F.normalize(Z,dim=-1), nid_map

# — training
for epoch in range(1,epochs_stage1+1):
    total=0.0; opt.zero_grad()

    # bills
    micro=0
    for batch in tqdm(bill_loader, desc=f'epoch {epoch} bills'):
        batch=batch.to(device); _,h2,_,_=forward_block(batch)
        loss= w_bill*bill_loss(batch,h2)
        (loss*(1.0/accum_bill)).backward(); total+=float(loss.item()); micro+=1
        if micro%accum_bill==0:
            nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
    if micro%accum_bill!=0:
        nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()

    # LT stance
    if stance_head_LT and lt_loader and (lt_topic_label is not None):
        micro=0
        for batch in tqdm(lt_loader, desc=f'epoch {epoch} LT-stance'):
            batch=batch.to(device); loss=stance_training_step(batch,'legislator_term',stance_head_LT,lt_topic_label,lt_topic_conf)
            if loss is not None:
                (loss*(1.0/accum_stance)).backward(); total+=float(loss.item()); micro+=1
                if micro%accum_stance==0:
                    nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
        if micro%accum_stance!=0:
            nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()

    # Contrast (cached)
    if pos_pairs:
        with torch.no_grad():
            Z, gid = compute_bv_embeddings(sample_frac=0.5, max_nodes=120_000)
            if Z is not None:
                rev = -torch.ones(data['bill_version'].num_nodes, dtype=torch.long)
                rev[gid] = torch.arange(gid.numel())
                pi = []; pj = []
                for a, b in pos_pairs:
                    ia, ib = rev[a].item(), rev[b].item()
                    if ia >= 0 and ib >= 0:
                        pi.append(ia); pj.append(ib)
                if len(pi) > 0:
                    pi = torch.tensor(pi, dtype=torch.long)
                    pj = torch.tensor(pj, dtype=torch.long)

        if 'Z' in locals() and Z is not None and 'pi' in locals() and pi.numel() > 0:
            Npairs = pi.numel()
            bsz = 4096
            micro = 0
            for s in range(0, Npairs, bsz):
                e = min(s + bsz, Npairs)
                ii = pi[s:e]       # indices into Z (CPU)
                jj = pj[s:e]

                # build negative pool and ensure positives are included
                pool = torch.randperm(Z.size(0), generator=rng, device=torch.device('cpu'))[:8192]
                pool = torch.unique(torch.cat([pool, ii, jj], dim=0))  # union on CPU

                # project ONLY the pool this micro-batch (fresh graph per micro)
                Zpool = F.normalize(proj_contrast(Z[pool].to(device)), dim=-1)  # [P, d]

                # map global indices -> pool positions
                revp = -torch.ones(Z.size(0), dtype=torch.long, device=device)
                revp[pool.to(device)] = torch.arange(pool.numel(), device=device)

                pos_i = revp[ii.to(device)]   # [B]
                pos_j = revp[jj.to(device)]   # [B]

                Zi = Zpool[pos_i]             # [B, d]
                logits = (Zi @ Zpool.T) / tau_contrast
                targets = pos_j

                loss_c = F.cross_entropy(logits, targets, label_smoothing=0.02) \
                        + lambda_l2 * _l2_mean(Zpool)

                (w_contrast * loss_c * (1.0/accum_contrast)).backward()
                total += float((w_contrast * loss_c).item())
                micro += 1
                if micro % accum_contrast == 0:
                    nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); opt.zero_grad()

            if micro % accum_contrast != 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()


    if torch.backends.mps.is_available(): torch.mps.synchronize()
    gc.collect(); sched.step(); print(epoch, round(total,4)); metrics.log('loss_stage1', total)

# — stage 2
h2_cache={'committee':None}; MAX_SNAPSHOTS_PER_ET=50; SNAP_PROB_PER_BATCH=0.01; MAX_EDGES_PER_SNAPSHOT=5000

def snapshot_alphas(batch,alphas):
    if random.random()>SNAP_PROB_PER_BATCH: return
    for et,a in alphas.items():
        if a is None: continue
        bucket=alpha_snapshots[str(et)]
        if len(bucket)>=MAX_SNAPSHOTS_PER_ET: continue
        take=min(MAX_EDGES_PER_SNAPSHOT, a.numel()); bucket.append(a.detach().cpu()[:take])

def route_batch(seq_h,lengths,y_true):
    if seq_h is None: return torch.tensor(0.0,device=device), None
    _,score=route_enc(seq_h,lengths); y=y_true[:score.size(0)].float()
    return F.binary_cross_entropy_with_logits(score,y), score

def build_routes_for_batch(batch,h2):
    if ('bill_version','is_version','bill') not in batch.edge_types: return None,None,None
    if ('bill_version','read','committee') not in batch.edge_types or 'committee' not in h2 or 'bill' not in batch.node_types: return None,None,None
    isv=batch[('bill_version','is_version','bill')].edge_index
    bv2b=torch.full((batch['bill_version'].num_nodes,), -1, device=device, dtype=torch.long); bv2b[isv[0]]=isv[1]
    eidx=batch[('bill_version','read','committee')].edge_index; eattr=batch[('bill_version','read','committee')].get('edge_attr',None)
    ts=pick_ts_from_cols(eattr, get_time_cols_for_edge(('bill_version','read','committee'))); ts=ts.to(device) if ts is not None else torch.zeros(eidx.size(1),device=device)
    b=bv2b[eidx[0]]; m=(b>=0)&(b<batch['bill'].batch_size); b=b[m]; cm=eidx[1][m]; t=ts[m]
    routes=[[] for _ in range(batch['bill'].batch_size)]
    for bi,ci,ti in zip(b.tolist(), cm.tolist(), t.tolist()): routes[bi].append((ti,ci))
    lengths=torch.tensor([len(r) for r in routes], device=device); maxL=int(lengths.max().item()) if lengths.numel()>0 else 0
    if maxL==0: return None,None,None
    seq=torch.zeros(batch['bill'].batch_size, maxL, hidden_dim, device=device)
    for bi,r in enumerate(routes):
        if not r: continue
        r=sorted(r,key=lambda x:x[0]); ids=[x[1] for x in r]; seq[bi,:len(ids)]=h2['committee'][ids]
    return seq,lengths,routes

def per_topic_metrics(probs,y,topics):
    out={}
    for t in range(max(K_topics,1)):
        m=(topics==t) if K_topics>0 else torch.ones_like(y).bool()
        if m.sum()<10: continue
        p=probs[m,1]; yt=y[m]; pred=(p>=0.5).long()
        tp=((pred==1)&(yt==1)).sum().item(); tn=((pred==0)&(yt==0)).sum().item().__int__()
        fp=((pred==1)&(yt==0)).sum().item(); fn=((pred==0)&(yt==1)).sum().item()
        acc=(tp+tn)/max(1,tp+tn+fp+fn); out[t]={'Acc':acc,'PosRate':float(p.mean().item()),'Count':int(m.sum().item())}
    return out

for epoch in range(epochs_stage1+1, epochs_stage1+epochs_stage2+1):
    total=0.0
    opt.zero_grad()
    micro=0
    for batch in tqdm(bill_loader, desc=f'epoch {epoch} inf+dose+route'):
        batch=batch.to(device)
        _,h2,alphas,_=forward_block(batch)
        if 'committee' in batch.node_types: h2_cache['committee']=h2['committee'].detach()
        n_b=batch['bill'].batch_size
        tid=(batch['bill'].cluster[:n_b].to(device).long() if (K_topics>0 and hasattr(batch['bill'],'cluster')) else torch.zeros(n_b,dtype=torch.long,device=device))
        logits2=bill_pred(h2['bill'][:n_b], tid); y2=batch['bill'].y_success[:n_b].to(device)
        loss_main=F.cross_entropy(logits2, y2, label_smoothing=0.02)

        s_full=F.softmax(logits2,dim=-1)[:,1].detach()
        loss_inf=torch.tensor(0.0,device=device)
        for actor_type in ['committee','donor','lobby_firm','legislator_term']:
            if actor_type not in batch.node_types: continue
            masks,actor_ids=actor_targeted_masks(batch,actor_type,max_actors=128)
            _,h22,_,_=forward_block(batch, edge_w_override=masks)
            masked_logits=bill_pred(h22['bill'][:n_b], tid); s_mask=F.softmax(masked_logits,dim=-1)[:,1].detach()
            margin=torch.relu(s_full-s_mask).mean(); loss_inf=loss_inf+margin
            update_influence_topic(batch,actor_type,actor_ids,s_full,s_mask,tid)
        loss_inf=loss_inf/4.0

        t_bins,topic_ids,qbins=build_treatment_bins_for_batch(batch,h2['bill'][:n_b],bins=bins_dose)
        if t_bins.size(0)>0:
            out_dr=dose_head(h2['bill'][:t_bins.size(0)], t_bins, topic_ids); y_bin=y2[:t_bins.size(0)].float()
            loss_dr=F.binary_cross_entropy_with_logits(out_dr,y_bin)
            byb=[]
            for b in range(t_bins.size(1)):
                m=(t_bins.argmax(1)==b); byb.append(out_dr[m].mean().unsqueeze(0) if m.any() else out_dr.new_zeros(1))
            mono=monotonic_penalty(torch.cat(byb,0).unsqueeze(0))
        else:
            loss_dr=torch.tensor(0.0,device=device); mono=torch.tensor(0.0,device=device)

        seq,lengths,routes=build_routes_for_batch(batch,h2)
        loss_route,r_score=route_batch(seq,lengths,y2)

        loss=1.0*loss_main + w_inf*loss_inf + w_dose*loss_dr + 0.05*mono + w_route*loss_route + lambda_l2*_l2_mean(h2['bill'][:n_b])
        (loss*(1.0/accum_bill)).backward(); total+=float(loss.item()); snapshot_alphas(batch,alphas)
        micro+=1
        if micro%accum_bill==0:
            nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
    if micro%accum_bill!=0:
        nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()

    # stance heads
    for (ldr, ntype, head, labs, conf, desc) in [
        (comm_loader, 'committee', stance_head_comm, comm_topic_label, comm_topic_conf, 'committee-stance'),
        (don_loader, 'donor', stance_head_donor, don_topic_label, don_topic_conf, 'donor-stance'),
        (lob_loader, 'lobby_firm', stance_head_lobby, (lob_lt_label if lob_lt_label is not None else lob_cm_label), (lob_lt_conf if lob_lt_conf is not None else lob_cm_conf), 'lobby-stance'),
    ]:
        if head is not None and ldr is not None and labs is not None:
            micro=0
            for batch in tqdm(ldr, desc=f'epoch {epoch} {desc}'):
                batch=batch.to(device); loss=stance_training_step(batch, ntype, head, labs, conf)
                if loss is not None and loss.requires_grad:
                    (loss*(1.0/accum_stance)).backward(); total+=float(loss.item()); micro+=1
                    if micro%accum_stance==0:
                        nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
            if micro%accum_stance!=0:
                nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()

    sched.step();
    if torch.backends.mps.is_available(): torch.mps.synchronize()
    print(epoch, round(total,4)); metrics.log('loss_stage2', total)

# — calibration
class BillHeadWrapper(nn.Module):
    def __init__(self, base): super().__init__(); self.base=base
    def forward(self,h,topic_ids): return self.base(h,topic_ids)

calibrated_topic_head=CalibratedPerTopic(BillHeadWrapper(bill_head_tc), K_topics if K_topics>0 else 1).to('cpu')
calibrated_topic_head.fit(lambda: collect_bill_logits_embeddings(bill_head_tc, val_bill_ids))
logits_all,targets_all,topics_all=collect_bill_logits_embeddings(bill_head_tc, val_bill_ids)

with torch.no_grad():
    logits_cal=[]
    for t in range(max(K_topics,1)):
        m=(topics_all==t) if K_topics>0 else torch.ones_like(targets_all).bool()
        lt=logits_all[m]
        if lt.numel()==0: continue
        lc=calibrated_topic_head.scalers[t](lt.to('cpu')); logits_cal.append((m,lc))
    probs_uncal=F.softmax(logits_all,dim=-1)
    ece_uncal=ece(probs_uncal,targets_all,n_bins=cal_bins); brier_uncal=brier(probs_uncal,targets_all)
    probs_cal=torch.zeros_like(probs_uncal)
    for m,lc in logits_cal: probs_cal[m]=F.softmax(lc,dim=-1)
    ece_cal=ece(probs_cal,targets_all,n_bins=cal_bins); brier_cal=brier(probs_cal,targets_all)
    per_topic_stat={}
    for t in range(max(K_topics,1)):
        m=(topics_all==t) if K_topics>0 else torch.ones_like(targets_all).bool()
        if m.sum()==0: continue
        p=probs_cal[m,1]; yt=targets_all[m]; pred=(p>=0.5).long()
        tp=((pred==1)&(yt==1)).sum().item(); tn=((pred==0)&(yt==0)).sum().item()
        fp=((pred==1)&(yt==0)).sum().item(); fn=((pred==0)&(yt==1)).sum().item()
        acc=(tp+tn)/max(1,tp+tn+fp+fn); per_topic_stat[t]={'Acc':acc,'PosRate':float(p.mean().item()),'Count':int(m.sum().item())}

# — attribution + export
def topk_actor_attributions_for_batch(batch, alphas, edge_index_dict, n_b, k=5):
    out=[[] for _ in range(n_b)]
    for et,a in alphas.items():
        if a is None: continue
        ei=edge_index_dict[et]; dst=ei[1]
        if et[2]=='bill':
            for i in range(ei.size(1)):
                b=int(dst[i].item());
                if b>=n_b: continue
                src=int(ei[0,i].item()); out[b].append((str(et),src,float(a[i].item())))
    topk=[]
    for b in range(n_b): topk.append(sorted(out[b], key=lambda x:x[2], reverse=True)[:k])
    return topk

def compute_version_drift():
    drift={}
    if ('bill_version','priorVersion','bill_version') not in data.edge_types: return drift
    ei=data[('bill_version','priorVersion','bill_version')].edge_index
    loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('bill_version', torch.arange(data['bill_version'].num_nodes)), batch_size=4096, shuffle=False)
    reps=torch.zeros(data['bill_version'].num_nodes, hidden_dim)
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device); _,h2,_,_=forward_block(batch); n=batch['bill_version'].batch_size
            reps[batch['bill_version'].n_id[:n].cpu()]=h2['bill_version'][:n].detach().cpu()
    z=F.normalize(reps,dim=-1)
    for i in range(ei.size(1)):
        a=int(ei[0,i]); b=int(ei[1,i]); d=float(1.0-(z[a]*z[b]).sum().item())
        drift.setdefault(int(bv_to_bill[b].item()), []).append({'v_prev':a,'v_next':b,'cosine_distance':d})
    return drift

export_payload={'bills':{},'actors':{},'topics':{},'meta':{}}

class InferenceBillIterator:
    def __iter__(self):
        return NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=('bill', torch.arange(data['bill'].num_nodes)), batch_size=1024, shuffle=False).__iter__()

def export_all():
    with torch.no_grad():
        for batch in InferenceBillIterator():
            batch=batch.to(device); _,h2,alphas,edge_idx=forward_block(batch)
            n_b=batch['bill'].batch_size
            tid=(batch['bill'].cluster[:n_b].to(device).long() if (K_topics>0 and hasattr(batch['bill'],'cluster')) else torch.zeros(n_b,dtype=torch.long,device=device))
            logits=bill_head_tc(h2['bill'][:n_b], tid); probs_unc=F.softmax(logits,dim=-1)
            logits_cal_local=torch.zeros_like(logits.cpu())
            for t in range(max(K_topics,1)):
                m=(tid.cpu()==t)
                if m.any(): logits_cal_local[m]=calibrated_topic_head.scalers[t](logits[m].cpu())
            probs_cal_local=F.softmax(logits_cal_local,dim=-1)
            topk=topk_actor_attributions_for_batch(batch, alphas, edge_idx, n_b, k=5)
            for i in range(n_b):
                gi=int(batch['bill'].n_id[i].item()) if hasattr(batch['bill'],'n_id') else int(i)
                tgi=int(tid[i].item()) if K_topics>0 else 0
                export_payload['bills'][gi]={
                    'P_success_uncalibrated_overall': float(probs_unc[i,1].item()),
                    'P_success_calibrated_overall': float(probs_cal_local[i,1].item()),
                    'topic_id': tgi,
                    'top_k_actors': [{'edge_type': et, 'src_id': int(s), 'score': float(sc)} for (et,s,sc) in topk[i]]
                }
        if K_topics>0:
            for typ,buf in influence_ema.items():
                if buf is None: continue
                z=(buf-buf.mean(0,keepdim=True))/(buf.std(0,keepdim=True)+1e-6); s=torch.sigmoid(z)
                export_payload['actors'].setdefault(typ,{})
                for i in range(buf.size(0)): export_payload['actors'][typ][int(i)]={'influence_score_by_topic': s[i].tolist()}
        for ntype,head,labs,conf in [
            ('legislator_term', stance_head_LT, lt_topic_label, lt_topic_conf),
            ('committee',       stance_head_comm, comm_topic_label, comm_topic_conf),
            ('donor',           stance_head_donor, don_topic_label, don_topic_conf),
            ('lobby_firm',      stance_head_lobby, (lob_lt_label if lob_lt_label is not None else lob_cm_label), (lob_lt_conf if lob_lt_conf is not None else lob_cm_conf)),
        ]:
            if head is None or labs is None or ntype not in data.node_types: continue
            loader=NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=(ntype, torch.arange(data[ntype].num_nodes)), batch_size=2048, shuffle=False)
            export_payload['actors'].setdefault(ntype,{})
            for batch in loader:
                batch=batch.to(device); _,h2,_,_=forward_block(batch); n=batch[ntype].batch_size; nid=batch[ntype].n_id[:n]
                logits=head(h2[ntype][:n]).view(-1,K_topics,3); probs=F.softmax(logits,dim=-1).detach().cpu()
                confs=(labs[nid.cpu()]>=0).float()*conf[nid.cpu()]
                for j in range(n):
                    gi=int(nid[j].item()); export_payload['actors'][ntype].setdefault(gi, {'stance':{}})
                    for t in range(K_topics):
                        p=probs[j,t].tolist(); c=float(confs[j,t].item())
                        export_payload['actors'][ntype][gi]['stance'][t]={'probs':p,'uncertainty':c}
        drift=compute_version_drift()
        for b,arr in drift.items():
            if b in export_payload['bills']: export_payload['bills'][b]['version_drift']=arr
        topic_stats={int(t):stat for t,stat in per_topic_stat.items()}
        export_payload['meta']={
            'ECE_uncal':round(ece_uncal,4),'Brier_uncal':round(brier_uncal,4),
            'ECE_cal':round(ece_cal,4),'Brier_cal':round(brier_cal,4),
            'per_topic_overall': topic_stats,
            'temperature_T_by_topic':[float(torch.exp(calibrated_topic_head.scalers[t].log_T.detach()).item()) for t in range(max(K_topics,1))],
            'alpha_snapshot_counts': {k:len(v) for k,v in alpha_snapshots.items()}
        }
        os.makedirs('outputs',exist_ok=True)
        with open('outputs/export_payload.json','w') as f: json.dump(export_payload,f)

export_all(); print("Export saved to outputs/export_payload.json")

model_state={
    'encoders':encoders.state_dict(),'proj':proj.state_dict(),'proj_dropout':proj_dropout.state_dict(),'norms':norms.state_dict(),
    'conv1':conv1.state_dict(),'conv2':conv2.state_dict(),'post_conv_dropout1':post_conv_dropout1.state_dict(),'post_conv_dropout2':post_conv_dropout2.state_dict(),
    'topic_emb': (topic_emb.state_dict() if topic_emb is not None else {}),'bill_head_tc':bill_head_tc.state_dict(),
    'route_enc':route_enc.state_dict(),'dose_head':dose_head.state_dict(),'proj_contrast':proj_contrast.state_dict()
}
if stance_head_LT is not None: model_state['stance_head_LT']=stance_head_LT.state_dict()
if stance_head_comm is not None: model_state['stance_head_comm']=stance_head_comm.state_dict()
if stance_head_donor is not None: model_state['stance_head_donor']=stance_head_donor.state_dict()
if stance_head_lobby is not None: model_state['stance_head_lobby']=stance_head_lobby.state_dict()
torch.save(model_state,'outputs/best_model_complete.pt')
torch.save({'scalers':[s.state_dict() for s in calibrated_topic_head.scalers], 'K_topics':K_topics}, 'outputs/calibrators.pt')
torch.save({'influence_ema':influence_ema, 'alpha_snapshots':{k:[a.shape[0] for a in v] for k,v in alpha_snapshots.items()}, 'export_payload':export_payload}, 'outputs/best_model_outputs.pt')
print("Saved model and outputs in outputs/")
