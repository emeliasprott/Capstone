import os, math, random, torch, numpy as np
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, HGTConv
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from tqdm import tqdm

# device, seeds, precision
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
gen_cpu = torch.Generator().manual_seed(42)
gen_dev = torch.Generator(device=device).manual_seed(42)
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.set_float32_matmul_precision('high')

CFG = dict(
    d=128, topics_expected=115, drop=0.15, heads=4, layers=3, lr=2e-3, wd=1e-4,
    vote_bsz=4096, bill_bsz=2048, eval_bsz=4096, time2vec_k=8, fp16=(device.startswith('cuda')),
    backbone='heterosage',  # 'hgt' or 'heterosage'
    alpha=1.0, beta=1.0, gamma=0.5, delta=0.5, eta=0.2, zeta=0.0, rho=0.01,
    tau=0.7, ls=0.05, ece_bins=15, shapley_K=256
)

def mlp(sizes, last_act=False):
    L=[];
    for i in range(len(sizes)-1):
        L += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2 or last_act: L += [nn.ReLU()]
    return nn.Sequential(*L)

class Time2Vec(nn.Module):
    def __init__(self, k=8): super().__init__(); self.w0=nn.Linear(1,1); self.wk=nn.Linear(1,k)
    def forward(self, t): t=t.view(-1,1); return torch.cat([self.w0(t), torch.sin(self.wk(t))], -1)

def add_topic_nodes(data: HeteroData):
    cl = data['bill'].cluster.long()
    mask = cl.ge(0)
    bill_ix = torch.nonzero(mask, as_tuple=False).view(-1)
    cl = cl[mask]
    uniq = sorted([int(u) for u in torch.unique(cl).tolist() if u!=-1])
    n_topics = len(uniq)
    data['topic'].num_nodes = n_topics
    remap = {t:i for i,t in enumerate(uniq)}
    t_for_bill = torch.full((data['bill'].num_nodes,), -1, dtype=torch.long)
    t_for_bill[bill_ix] = torch.tensor([remap[int(t)] for t in cl.tolist()], dtype=torch.long)
    ei = torch.stack([t_for_bill[bill_ix], bill_ix], 0)
    data[('topic','has','bill')].edge_index = ei
    data[('topic','has','bill')].edge_attr = torch.ones(ei.size(1),1)
    data['bill'].topic_ix = t_for_bill
    return n_topics, torch.tensor(uniq), t_for_bill

def edge_time2vec_for_batch(batch: HeteroData, t2v: Time2Vec):
    edge_t2v={}
    dev = next(t2v.parameters()).device
    for et in batch.edge_types:
        E = batch[et]
        if hasattr(E,'edge_time') and E.edge_time is not None:
            dt = E.edge_time.float().to(dev)
        elif hasattr(E,'edge_attr') and E.edge_attr is not None and E.edge_attr.size(-1)>=1:
            dt = E.edge_attr[:,-1].float().to(dev)
        else:
            dt = torch.zeros(E.edge_index.size(1), device=dev)
        edge_t2v[et] = t2v(dt)
    return edge_t2v

class Projections(nn.Module):
    def __init__(self, dims, d=128):
        super().__init__(); self.P=nn.ModuleDict({
            'bill': nn.Linear(dims.get('bill',770), d),
            'bill_version': nn.Linear(dims.get('bill_version',390), d),
            'legislator': nn.Linear(dims.get('legislator',385), d),
            'legislator_term': mlp([dims.get('legislator_term',4),64,d]),
            'committee': mlp([dims.get('committee',65),128,d]),
            'lobby_firm': nn.Linear(dims.get('lobby_firm',384), d),
            'donor': mlp([dims.get('donor',64),128,d]),
        })
    def forward(self, xdict): return {nt: self.P[nt](x) for nt,x in xdict.items() if nt in self.P}

class HeteroSAGE(nn.Module):
    def __init__(self, metadata, d=128, layers=3, drop=0.15, edge_t2v_dim=9):
        super().__init__(); self.drop=nn.Dropout(drop)
        self.edge_mlps = nn.ModuleDict({str(et): mlp([edge_t2v_dim, d]) for et in metadata[1]})
        convs=[]
        for _ in range(layers):
            rels={et: SAGEConv((d,d), d) for et in metadata[1]}
            convs.append(HeteroConv(rels, aggr='sum'))
        self.convs = nn.ModuleList(convs)
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(d) for nt in metadata[0]})
    def forward_with_edge_attrs(self, h, data, edge_t2v):
        for conv in self.convs:
            h = conv(h, {et: data[et].edge_index for et in data.edge_types},
                     {et: self.edge_mlps[str(et)](edge_t2v[et]) for et in data.edge_types})
            h = {k: self.norm[k](self.drop(v)) for k,v in h.items()}
        return h

class HGTBackbone(nn.Module):
    def __init__(self, metadata, d=128, layers=3, heads=4, drop=0.15, edge_t2v_dim=9):
        super().__init__(); self.edge_lin=nn.ModuleDict({str(et): nn.Linear(edge_t2v_dim, d) for et in metadata[1]})
        self.convs = nn.ModuleList([HGTConv(d, d, metadata, heads=heads, group='sum', dropout=drop, edge_dim=d) for _ in range(layers)])
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(d) for nt in metadata[0]}); self.drop=nn.Dropout(drop)
    def forward_with_edge_attrs(self, h, data, edge_t2v):
        edge_attr = {et: self.edge_lin[str(et)](edge_t2v[et]) for et in data.edge_types}
        for conv in self.convs:
            h = conv(h, {et:data[et].edge_index for et in data.edge_types}, edge_attr)
            h = {k: self.norm[k](self.drop(v)) for k,v in h.items()}
        return h

class MetaPathBlock(nn.Module):
    def __init__(self, d=128, routes=5): super().__init__(); self.fuse=mlp([routes*d, d]); self.alphas=nn.Parameter(torch.zeros(routes))
    def forward(self, h):
        z = torch.cat([h['topic'].mean(0), h['bill'].mean(0), h['bill_version'].mean(0), h['legislator_term'].mean(0), h['committee'].mean(0)], -1).unsqueeze(0)
        return self.fuse(z) * torch.softmax(self.alphas,0).sum()

class CrossAttentionLT2Bill(nn.Module):
    def __init__(self, d=128, h=4):
        super().__init__()
        self.h=h
        self.dk=d//h
        self.Wq=nn.Linear(d,d)
        self.Wk=nn.Linear(d,d)
        self.Wv=nn.Linear(d,d)
        self.out=nn.Linear(d,d)
    def forward(self, q_lt, k_bill):
        Q=self.Wq(q_lt).view(-1,self.h,self.dk); K=self.Wk(k_bill).view(-1,self.h,self.dk); V=self.Wv(k_bill).view(-1,self.h,self.dk)
        a=torch.softmax((Q*K).sum(-1)/math.sqrt(self.dk), -1).unsqueeze(-1)
        return self.out((a*V).reshape(-1, self.h*self.dk))

class VoteHead(nn.Module):
    def __init__(self,d=128): super().__init__(); self.m=mlp([4*d,2*d,d]); self.o=nn.Linear(d,3)
    def forward(self,hlt,hbv,htop,ctxt): return self.o(self.m(torch.cat([hlt,hbv,htop,ctxt],-1)))
class OutcomeHead(nn.Module):
    def __init__(self,d=128): super().__init__(); self.m=mlp([2*d,d]); self.o=nn.Linear(d,3)
    def forward(self,hbill,route): return self.o(self.m(torch.cat([hbill,route],-1)))
class GateHead(nn.Module):
    def __init__(self,d=128,k=6): super().__init__(); self.m=mlp([3*d,d]); self.o=nn.Linear(d,k)
    def forward(self,hc,hb,ht): return self.o(self.m(torch.cat([hc,hb,ht],-1)))
class StanceHead(nn.Module):
    def __init__(self,d=128): super().__init__(); self.m=mlp([2*d,d]); self.o=nn.Linear(d,1)
    def forward(self,ha,ht): return torch.tanh(self.o(self.m(torch.cat([ha,ht],-1))))

class MaskNet(nn.Module):
    def __init__(self,d=128): super().__init__(); self.m=mlp([d,d,1])
    def forward(self,h): return torch.sigmoid(self.m(h))
class MonoSpline(nn.Module):
    def __init__(self, knots=8): super().__init__(); self.w=nn.Parameter(torch.zeros(knots).uniform_(0,0.1)); self.b=nn.Parameter(torch.tensor(0.0))
    def forward(self,a): x=torch.stack([torch.clamp(a-k/8.0,min=0) for k in range(8)],-1); w=torch.cumsum(F.softplus(self.w),0); return self.b + (x*w).sum(-1,keepdim=True)

def orthogonality(E): Q=F.normalize(E,-1); G=Q@Q.t(); I=torch.eye(G.size(0), device=G.device); return ((G-I)**2).mean()
def expected_calibration_error(logits,y,bins=15):
    p=F.softmax(logits,-1).max(-1).values; pred=logits.argmax(-1); acc=(pred==y).float(); ece=0.0
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins; m=(p>=lo)&(p<hi)
        if m.any(): ece += (m.float().mean() * (acc[m].mean()-p[m].mean()).abs()).item()
    return torch.tensor(ece, device=logits.device)
def ce_balanced(logits,y,nclass=3,ls=0.05):
    y=y.long(); m=y.ge(0)
    if m.sum()==0: return logits.sum()*0
    y=y[m]; p=F.log_softmax(logits[m],-1)
    with torch.no_grad():
        f=torch.bincount(y, minlength=nclass).float().clamp_min(1); w=(f.sum()/f); w=w/w.mean()
    oh=F.one_hot(y,nclass).float(); oh=(1-ls)*oh+ls/nclass
    return -(w[y]*(oh*p).sum(-1)).mean()
def brier(logits,y):
    y=y.long(); m=y.ge(0)
    if m.sum()==0: return logits.sum()*0
    p=F.softmax(logits[m],-1); oh=F.one_hot(y[m],3).float()
    return ((p-oh)**2).mean()
def info_nce(q,k,tau=0.7):
    q=F.normalize(q,-1); k=F.normalize(k,-1); sim=q@k.t(); lab=torch.arange(q.size(0), device=q.device); return F.cross_entropy(sim/tau, lab)

def attentive_version_pool(batch: HeteroData, h, d=128):
    if ('bill_version','is_version','bill') not in batch.edge_types: return h['bill']
    ei = batch[('bill_version','is_version','bill')].edge_index
    bv = h['bill_version']; b = h['bill']
    agg = torch.zeros_like(b); cnt = torch.zeros(b.size(0), device=b.device).unsqueeze(-1)+1e-6
    agg.index_add_(0, ei[1], bv[ei[0]]); cnt.index_add_(0, ei[1], torch.ones_like(cnt).index_select(0, ei[1]))
    return agg/cnt

def build_topic_stance_labels(data: HeteroData, t_for_bill, min_eff=5):
    if ('legislator_term','voted_on','bill_version') not in data.edge_types: return None
    e = data[('legislator_term','voted_on','bill_version')]
    y = e.edge_attr[:, 0].long()
    bv2b = data[('bill_version','is_version','bill')].edge_index[1]
    b = bv2b[e.edge_index[1]]
    t = t_for_bill[b]
    m = y.ne(0) & t.ge(0)
    lt = e.edge_index[0][m]; yv = y[m]; tt = t[m]
    key = lt*1000 + tt
    size = data['legislator_term'].num_nodes*1000 + (0 if tt.numel()==0 else int(tt.max())+1)
    vals = torch.zeros(size); cnts = torch.zeros(size)
    vals.index_add_(0, key, yv.float()); cnts.index_add_(0, key, torch.ones_like(yv, dtype=torch.float))
    stance = torch.zeros(data['legislator_term'].num_nodes, 0 if tt.numel()==0 else int(tt.max())+1)
    eff = torch.zeros_like(stance)
    if tt.numel()>0:
        idx_lt = (key//1000).long(); idx_t = (key%1000).long()
        stance = torch.zeros(data['legislator_term'].num_nodes, int(tt.max())+1); eff = torch.zeros_like(stance)
        stance[idx_lt, idx_t] = vals[key]/cnts[key].clamp_min(1); eff[idx_lt, idx_t] = cnts[key]
        stance = torch.where(eff>=min_eff, stance, torch.nan*stance)
    return stance

class Capstone(nn.Module):
    def __init__(self, data: HeteroData, cfg=CFG):
        super().__init__()
        self.cfg=cfg
        d=cfg['d']
        data = ToUndirected()(data)
        data = RemoveIsolatedNodes()(data)
        dims={nt: (data[nt].x.size(-1) if hasattr(data[nt],'x') and data[nt].x is not None else 0) for nt in data.node_types}
        self.proj=Projections(dims,d)
        self.topic_emb=nn.Embedding(getattr(data['topic'],'num_nodes',1), d)
        self.t2v=Time2Vec(cfg['time2vec_k']); self.drop=nn.Dropout(cfg['drop'])
        self.backbone = HeteroSAGE(data.metadata(), d, cfg['layers'], cfg['drop']) if cfg['backbone']=='heterosage' else HGTBackbone(data.metadata(), d, cfg['layers'], cfg['heads'], cfg['drop'])
        self.metapath=MetaPathBlock(d,5); self.xattn=CrossAttentionLT2Bill(d)
        self.vote_head=VoteHead(d); self.outcome_head=OutcomeHead(d); self.gate_head=GateHead(d,6)
        self.stance_lt=StanceHead(d); self.stance_donor=StanceHead(d); self.stance_lobby=StanceHead(d)
        self.mask_actor=MaskNet(d); self.mask_comm=MaskNet(d); self.dose_spline=MonoSpline(8)
        self.temp=nn.Parameter(torch.tensor(1.0))
    def encode(self, batch: HeteroData):
        edge_t2v = edge_time2vec_for_batch(batch, self.t2v)
        x={nt: (batch[nt].x if hasattr(batch[nt],'x') and batch[nt].x is not None else torch.zeros(getattr(batch[nt],'num_nodes',1), self.cfg['d'], device=batch[nt].__dict__.get('x', torch.empty(0, device=device)).device)) for nt in batch.node_types}
        h=self.proj({k:v for k,v in x.items() if k!='topic'}); h['topic']=self.topic_emb.weight
        h=self.backbone.forward_with_edge_attrs(h, batch, edge_t2v)
        pooled=attentive_version_pool(batch,h,self.cfg['d']); h['bill']=0.5*h['bill']+0.5*pooled
        return h
    def route_encoding(self,h):
        z=torch.cat([h['topic'].mean(0),h['bill'].mean(0),h['bill_version'].mean(0),h['legislator_term'].mean(0),h['committee'].mean(0)],-1).unsqueeze(0)
        return self.metapath(h).expand(h['bill'].size(0), -1)
    def vote_logits(self,batch,h,e_idx,t_for_bill):
        lt,bv=e_idx; b=batch[('bill_version','is_version','bill')].edge_index[1][bv]; t=t_for_bill[b]
        hlt=h['legislator_term'][lt]; hbv=h['bill_version'][bv]; htop=h['topic'][t]; ctxt=self.xattn(hlt,h['bill'][b])
        return self.vote_head(hlt,hbv,htop,ctxt)
    def outcome_logits(self,h,route): return self.outcome_head(h['bill'],route)
    def gate_logits(self,batch,h,e_idx,t_for_bill):
        bv,c=e_idx; b=batch[('bill_version','is_version','bill')].edge_index[1][bv]; t=t_for_bill[b]
        return self.gate_head(h['committee'][c], h['bill'][b], h['topic'][t])
    def stance_pred(self,h,actor_type,actor_idx,topic_idx):
        if actor_type=='legislator_term': return self.stance_lt(h['legislator_term'][actor_idx], h['topic'][topic_idx])
        if actor_type=='donor': return self.stance_donor(h['donor'][actor_idx], h['topic'][topic_idx])
        if actor_type=='lobby_firm': return self.stance_lobby(h['lobby_firm'][actor_idx], h['topic'][topic_idx])
        raise ValueError

def build_loaders(data: HeteroData, cfg=CFG):
    if hasattr(data[('legislator_term','voted_on','bill_version')],'edge_label'):
        el=data[('legislator_term','voted_on','bill_version')].edge_label
        if hasattr(el,'device') and str(el.device)!='cpu':
            data[('legislator_term','voted_on','bill_version')].edge_label = el.cpu()
    vote_loader = NeighborLoader(
        data,
        input_nodes=('legislator_term', torch.arange(data['legislator_term'].num_nodes)),
        num_neighbors={
            ('legislator_term','voted_on','bill_version'):[64,64,64],
            ('bill_version','is_version','bill'):[8,8,8],
            ('bill_version', 'priorVersion', 'bill_version'): [4,4,4],
            ('bill_version','read','committee'):[6,6,6],
            ('legislator_term','member_of','committee'):[8,8,8],
            ('legislator', 'samePerson', 'legislator_term'): [4,4,4],
            ('topic','has','bill'):[16,16,16],
            ('legislator_term','wrote','bill_version'):[6,6,6],
            ('donor','donated_to','legislator_term'):[16,16,16],
            ('lobby_firm','lobbied','legislator_term'):[16,16,16],
            ('lobby_firm','lobbied','committee'):[16,16,16]
        },
        batch_size=cfg['vote_bsz'], shuffle=True,
        num_workers=max(2, os.cpu_count()//2), pin_memory=True, persistent_workers=True
    )
    bill_loader = NeighborLoader(
        data,
        input_nodes=('bill', torch.arange(data['bill'].num_nodes)),
        num_neighbors={
            ('bill_version','is_version','bill'):[8,8,8],
            ('bill_version','read','committee'):[8,8,8],
            ('legislator_term','voted_on','bill_version'):[32,32,32],
            ('topic','has','bill'):[16,16,16]
        },
        batch_size=cfg['bill_bsz'], shuffle=True,
        num_workers=max(2, os.cpu_count()//2), pin_memory=True, persistent_workers=True
    )
    return vote_loader, bill_loader

def macro_f1(logits,y):
    y=y.long(); m=y.ge(0)
    if m.sum()==0: return torch.tensor(0.0, device=logits.device)
    y,yh=y[m], logits[m].argmax(-1); res=0.0
    for c in range(3):
        tp=((yh==c)&(y==c)).sum().float(); fp=((yh==c)&(y!=c)).sum().float(); fn=((yh!=c)&(y==c)).sum().float()
        p=tp/(tp+fp+1e-6); r=tp/(tp+fn+1e-6); f=2*p*r/(p+r+1e-6); res+=f
    return res/3

class Trainer:
    def __init__(self, data: HeteroData, cfg=CFG):
        self.data = data  # CPU graph for samplers
        n_topics, topic_vals, t_for_bill = add_topic_nodes(self.data)
        assert n_topics==CFG['topics_expected'], f"Expected {CFG['topics_expected']} topics, got {n_topics}"
        self.t_for_bill = t_for_bill
        self.model = Capstone(self.data, cfg).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        self.vote_loader, self.bill_loader = build_loaders(self.data, cfg)
        self.cfg=cfg; self.topic_vals=topic_vals
        self.stance_labels = build_topic_stance_labels(self.data, self.t_for_bill, min_eff=5)
    def train_epoch(self):
        self.model.train()
        tot={'vote':0.0,'out':0.0,'gate':0.0,'stance':0.0,'contrast':0.0,'reg':0.0}
        for batch in tqdm(self.vote_loader, desc='vote'):
            batch = batch.to(device, non_blocking=True)
            h = self.model.encode(batch)
            e = batch[('legislator_term','voted_on','bill_version')]
            logits = self.model.vote_logits(batch,h,e.edge_index, batch['bill'].topic_ix if hasattr(batch['bill'],'topic_ix') else self.t_for_bill.to(device))
            y = e.edge_label.to(device) if hasattr(e,'edge_label') else torch.zeros(logits.size(0),dtype=torch.long,device=device)
            L_vote = ce_balanced(logits,y,ls=self.cfg['ls']) + 0.1*brier(logits,y)
            q = h['bill'][batch['bill'].n_id] if hasattr(batch['bill'],'n_id') else h['bill']
            kpos = h['topic'][(batch['bill'].topic_ix[batch['bill'].n_id]).clamp_min(0)] if hasattr(batch['bill'],'n_id') and hasattr(batch['bill'],'topic_ix') else h['topic'][(self.t_for_bill.to(device)).clamp_min(0)]
            kneg = kpos[torch.randperm(kpos.size(0), generator=gen_dev, device=device)]
            L_con = info_nce(q,kpos,self.cfg['tau']) + info_nce(q,kneg,self.cfg['tau'])
            L_reg = 0.001*orthogonality(self.model.topic_emb.weight)
            L = self.cfg['alpha']*L_vote + self.cfg['eta']*L_con + self.cfg['rho']*L_reg
            self.opt.zero_grad(); L.backward(); self.opt.step()
            tot['vote'] += float(L_vote.item()); tot['contrast'] += float(L_con.item()); tot['reg'] += float(L_reg.item())
        for batch in tqdm(self.bill_loader, desc='outcome'):
            batch = batch.to(device, non_blocking=True)
            h = self.model.encode(batch)
            route = self.model.metapath(h).expand(h['bill'].size(0), -1)
            logits = self.model.outcome_logits(h, route)
            y = batch['bill'].y.to(device) if hasattr(batch['bill'],'y') else torch.zeros(logits.size(0),dtype=torch.long,device=device)
            L_out = ce_balanced(logits,y,ls=self.cfg['ls']) + 0.1*brier(logits,y)
            self.opt.zero_grad(); L_out.backward(); self.opt.step()
            tot['out'] += float(L_out.item())
        if ('bill_version','read','committee') in self.data.edge_types:
            e = self.data[('bill_version','read','committee')]
            idx = torch.randperm(e.edge_index.size(1), generator=gen_cpu)[:self.cfg['eval_bsz']]
            sub = self.data
            sub = sub.to(device, non_blocking=True)
            h = self.model.encode(sub)
            logits = self.model.gate_logits(sub,h,e.edge_index[:,idx.to(device)], (sub['bill'].topic_ix if hasattr(sub['bill'],'topic_ix') else self.t_for_bill.to(device)))
            y = getattr(e,'stage', torch.zeros(idx.size(0), dtype=torch.long)).to(device).clamp(0,5)
            L_gate = ce_balanced(logits,y,nclass=6,ls=self.cfg['ls'])
            self.opt.zero_grad(); L_gate.backward(); self.opt.step()
            tot['gate'] += float(L_gate.item())
        if self.stance_labels is not None and self.stance_labels.numel()>0:
            nlt, nt = self.stance_labels.size()
            lt = torch.randint(0,nlt,(self.cfg['eval_bsz'],), generator=gen_cpu)
            t = torch.randint(0,nt,(self.cfg['eval_bsz'],), generator=gen_cpu)
            full = self.data.to(device, non_blocking=True)
            h = self.model.encode(full)
            pred = self.model.stance_pred(h,'legislator_term',lt.to(device),t.to(device)).squeeze(1)
            target = self.stance_labels[lt,t].to(device)
            mask = torch.isfinite(target)
            if mask.any():
                L_st = F.mse_loss(pred[mask], target[mask])
                self.opt.zero_grad(); L_st.backward(); self.opt.step()
                tot['stance'] += float(L_st.item())
        return tot
    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval(); m={'vote_f1':0.0,'vote_ece':0.0,'out_f1':0.0,'out_ece':0.0}; acc=0
        for batch in self.vote_loader:
            batch=batch.to(device, non_blocking=True); h=self.model.encode(batch)
            e=batch[('legislator_term','voted_on','bill_version')]
            logits=self.model.vote_logits(batch,h,e.edge_index, batch['bill'].topic_ix if hasattr(batch['bill'],'topic_ix') else self.t_for_bill.to(device))
            y=e.edge_label.to(device)
            m['vote_f1'] += float(macro_f1(logits,y)); m['vote_ece'] += float(expected_calibration_error(logits[y>=0], y[y>=0], bins=self.cfg['ece_bins'])); acc+=1
            if acc>=5: break
        acc=0
        for batch in self.bill_loader:
            batch=batch.to(device, non_blocking=True); h=self.model.encode(batch)
            route=self.model.metapath(h).expand(h['bill'].size(0), -1); logits=self.model.outcome_logits(h,route); y=batch['bill'].y.to(device)
            m['out_f1'] += float(macro_f1(logits,y)); m['out_ece'] += float(expected_calibration_error(logits[y>=0], y[y>=0], bins=self.cfg['ece_bins'])); acc+=1
            if acc>=5: break
        for k in m: m[k]/=max(1,acc)
        return m
    @torch.no_grad()
    def embed_full(self):
        self.model.eval()
        full = self.data.to(device, non_blocking=True)
        return self.model.encode(full)
    @torch.no_grad()
    def influence_all(self, h):
        out_actor_topic=[]; out_actor_overall=[]; out_comm=[]
        nlt=self.data['legislator_term'].num_nodes; nt=self.data['topic'].num_nodes
        E=torch.rand(nt, device=h['topic'].device); Z=torch.rand(nt, device=h['topic'].device); R=torch.rand(nt, device=h['topic'].device); C=torch.full((nt,),0.8, device=h['topic'].device)
        for a in range(nlt):
            t_idx=torch.arange(nt, device=h['topic'].device)
            S=self.model.stance_pred(h,'legislator_term',torch.full((nt,),a,device=h['topic'].device),t_idx).squeeze(1)
            I=torch.zeros_like(S)
            def norm(x): x=(x-x.mean())/(x.std()+1e-6); return torch.sigmoid(x)
            score=(norm(E)+norm(Z)+norm(S.abs())+norm(I)+norm(R))*norm(C); vals, tidx=torch.topk(score, min(5,score.numel()))
            out_actor_overall.append({'actor_id':int(a),'actor_type':'legislator_term','overall_influence':float((I*E).mean().item()),'ci_lo':0.0,'ci_hi':0.0,
                                      'topic_breakdown':[{'topic_id':int(i), 'weight':float(E[int(i)].item()), 'delta':float(I[int(i)].item())} for i in tidx.tolist()],
                                      'top_topics':[int(i) for i in tidx.tolist()]})
            for i in range(nt):
                out_actor_topic.append({'actor_id':int(a),'actor_type':'legislator_term','topic_id':int(i),
                                        'stance':float(S[i].item()),'stance_ci_lo':float(S[i].item()-0.1),'stance_ci_hi':float(S[i].item()+0.1),
                                        'influence_delta_mean':float(I[i].item()),'influence_ci_lo':0.0,'influence_ci_hi':0.0,
                                        'engagement':float(E[i].item()),'salience':float(Z[i].item()),'recency':float(R[i].item()),'certainty':float(C[i].item()),
                                        'topness_score':float(vals.mean().item()),
                                        'pathway_share':{'vote_share':1.0,'committee_share':0.0},
                                        'support':{'n_votes':0,'n_final':0,'n_comm_reads':0,'spend':0.0,'lobby_touches':0},
                                        'evidence':{'top_paths':[],'pivotal_bills':[]}})
        for c in range(self.data['committee'].num_nodes):
            out_comm.append({'committee_id':int(c),'overall_influence':0.0,'ci_lo':0.0,'ci_hi':0.0,'topic_breakdown':[],'top_topics':[],'gate_index':0.0})
        return {'actor_topic':out_actor_topic,'actor_overall':out_actor_overall,'committee_overall':out_comm}

def build_outputs(model: Capstone, data: HeteroData, h, rep):
    outs={'actor_topic':rep['actor_topic'],'actor_overall':rep['actor_overall'],'committee_overall':rep['committee_overall']}
    with torch.no_grad():
        route=model.metapath(h).expand(h['bill'].size(0), -1); logits=model.outcome_logits(h,route); P=F.softmax(logits,-1)
        outs['per_bill']=[{'bill_id':int(i),'P(pass)':float(P[i,2].item()),'P(veto)':float(P[i,1].item()),'P(fail)':float(P[i,0].item()),
                           'expected_margin':0.0,'pivotal_actors':[],'committee_bottlenecks':[]} for i in range(P.size(0))]
    return outs

def run_full_training(data: HeteroData, epochs=3):
    trainer=Trainer(data, CFG)
    for ep in range(epochs):
        losses=trainer.train_epoch(); metrics=trainer.eval_epoch()
        print(f'epoch {ep}:', {k:round(v,4) for k,v in losses.items()}, {k:round(v,4) for k,v in metrics.items()})
    h=trainer.embed_full(); rep=trainer.influence_all(h); outs=build_outputs(trainer.model, trainer.data, h, rep)
    return trainer, h, outs

if __name__=='__main__':
    data = torch.load('data4.pt', weights_only=False)
    trainer,h,outs = run_full_training(data, epochs=3)

