import datetime, gc, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.loader import HGTLoader

hidden_dim = 192
n_layers = 3
dropout_p = 0.10
device = torch.device('mps')

# utilities
def _init_linear(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=0.01)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def sanitize(t, clamp=1e4):
    t = t.float() if t.dtype == torch.float64 else t
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
    return t.clamp_(-clamp, clamp)

def _global_to_local(sorted_global, query):
    pos = torch.searchsorted(sorted_global, query)
    return pos

def alarm():
    import sounddevice as sd
    import time

    fs = 25500
    duration = 0.6
    frequency = 435

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = 1.05 * np.sin(2 * np.pi * frequency * t)

    for j in range(3):
        for i in range(3):
            sd.play(x, fs)
            sd.wait()

        time.sleep(0.5)

# preprocessing
def safe_normalize_timestamps(timestamps):
    timestamps = torch.nan_to_num(timestamps, nan=0.0, posinf=1e4, neginf=-1e4)
    min_time = timestamps.min()
    max_time = timestamps.max()
    if (max_time - min_time) < 1e-4:
        return torch.zeros_like(timestamps)
    return (timestamps - min_time) / (max_time - min_time)

def safe_standardize_time_format(time_data):
    times = []
    for t in time_data:
        try:
            if isinstance(t, (int, float)) and 1900 <= t  and t <= 2100:
                td = datetime.datetime(int(t), 6, 15).timestamp()
            elif (isinstance(t, str) or (isinstance(t, float))) and (float(t) < 2100 and float(t) > 1900):
                td = datetime.datetime(int(float(t)), 6, 15).timestamp()
            elif float(t) > 0 and float(t) < 1990:
                td = t
            elif float(t) > 17000000.0:
                td = float(t)
            elif isinstance(t, datetime.datetime):
                td = t.timestamp()
            else:
                td = float(t) * 1e9
        except:
            td = datetime.datetime(2000, 6, 15).timestamp()
        times.append(td)
    return torch.tensor(times, dtype=torch.float32)

def pull_timestamps(data):
    timestamp_edges = [
        ('donor', 'donated_to', 'legislator_term'),
        ('legislator_term', 'rev_donated_to', 'donor'),
        ('lobby_firm', 'lobbied', 'legislator_term'),
        ('lobby_firm', 'lobbied', 'committee'),
        ('committee', 'rev_lobbied', 'lobby_firm'),
        ('legislator_term', 'rev_lobbied', 'lobby_firm'),
        ('bill_version', 'rev_voted_on', 'legislator_term'),
        ('legislator_term', 'voted_on', 'bill_version'),
    ]
    timestamp_nodes = ['legislator_term', 'bill_version', 'bill']

    for et in timestamp_edges:
        if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None and len(data[et].edge_attr.size()) > 1:
            if data[et].edge_attr.size(1) > 1:
                edge_attr = data[et].edge_attr
                ts_col = edge_attr[:, -1]
                if ts_col.abs().max() > 1e8 or ts_col.min() < 0:
                    ts_col = safe_standardize_time_format(ts_col.tolist()).to(edge_attr.device)
                data[et].timestamp = safe_normalize_timestamps(ts_col)
                data[et].edge_attr = edge_attr[:, :-1]

    for nt in timestamp_nodes:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            try:
                if len(data[nt].x.size()) > 1:
                    if data[nt].x.size(1) > 1:
                        x = data[nt].x
                        ts_col = x[:, -1]
                        if ts_col.abs().max() > 1e8 or ts_col.min() < 0:
                            ts_col = safe_standardize_time_format(ts_col.tolist()).to(x.device)
                        if nt in timestamp_nodes or ts_col.abs().max() > 1e6:
                            data[nt].timestamp = safe_normalize_timestamps(ts_col)
                            data[nt].x = x[:, :-1]
            except:
                pass
    return data
def clean_features(data):
    for nt in data.node_types:
        x = data[nt].x
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True).clamp(min=1e-5)
        x = ((x - mean) / std).clamp(-10, 10)
        data[nt].x = x
        data[nt].x_mean = mean
        data[nt].x_std = std
    data = pull_timestamps(data)
    return data

def compute_controversiality(data):
    edge_type = ('legislator_term', 'voted_on', 'bill_version')
    if edge_type not in data.edge_index_dict:
        raise ValueError("Missing 'voted_on' edges in data.")

    ei = data[edge_type].edge_index
    ea = data[edge_type].edge_attr

    vote_signal = ea[:, 0]

    src_nodes = ei[0]
    tgt_nodes = ei[1]

    num_bills = data['bill_version'].num_nodes
    device = tgt_nodes.device

    yes_votes = torch.zeros(num_bills, device=device)
    no_votes = torch.zeros(num_bills, device=device)

    yes_votes.index_add_(0, tgt_nodes, (vote_signal > 0).float())
    no_votes.index_add_(0, tgt_nodes, (vote_signal < 0).float())

    total_votes = yes_votes + no_votes + 1e-6

    yes_ratio = yes_votes / total_votes
    no_ratio = no_votes / total_votes

    controversy = 4 * yes_ratio * no_ratio
    controversy = controversy.clamp(0, 1)
    data['bill_version'].controversy = controversy

    return data

def load_and_preprocess_data(path='data3.pt'):
    full_data = torch.load(path, weights_only=False)
    for nt in full_data.node_types:
        if hasattr(full_data[nt], 'x') and full_data[nt].x is not None:
            flat = torch.as_tensor(full_data[nt].x).flatten(start_dim=1)
            full_data[nt].x = flat
            full_data[nt].num_nodes = flat.size(0)

    for edge_type, edge_index in full_data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        max_src_idx = edge_index[0].max().item() if edge_index.size(1) > 0 else -1
        max_dst_idx = edge_index[1].max().item() if edge_index.size(1) > 0 else -1
        if max_src_idx >= full_data[src_type].num_nodes:
            print(f"Fixing {src_type} node count: {full_data[src_type].num_nodes} -> {max_src_idx + 1}")
            full_data[src_type].num_nodes = max_src_idx + 1

        if max_dst_idx >= full_data[dst_type].num_nodes:
            print(f"Fixing {dst_type} node count: {full_data[dst_type].num_nodes} -> {max_dst_idx + 1}")
            full_data[dst_type].num_nodes = max_dst_idx + 1
    full_data['bill'].y[np.where(full_data['bill'].y < 0)[0]] = 0

    data = ToUndirected(merge=False)(full_data)
    del full_data
    gc.collect()
    data = RemoveIsolatedNodes()(data)
    data = compute_controversiality(clean_features(data))

    for nt in data.node_types:
        ids = torch.arange(data[nt].num_nodes, device=device)
        data[nt].node_id = ids
    for store in data.stores:
        for key, value in store.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                store[key] = value.float()
    return data

# encoders
class Time2Vec(nn.Module):
    def __init__(self, d=12):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(()))
        self.w = nn.Parameter(torch.randn(d - 1))
        self.b = nn.Parameter(torch.randn(d))
    def forward(self, t):
        v0 = self.w0 * t + self.b[0]
        v = torch.sin(self.w * t.unsqueeze(-1) + self.b[1:])
        return torch.cat([v0.unsqueeze(-1), v], -1)

class FeatureProjector(nn.Module):
    def __init__(self, metadata, in_dims, d_out=hidden_dim, t_dim=12):
        super().__init__()
        self.t2v = Time2Vec(t_dim)
        self.prj = nn.ModuleDict()
        for nt in metadata[0]:
            use_t = nt in ['legislator_term', 'bill', 'bill_version']
            d_in  = in_dims[nt] + (t_dim if use_t else 0)
            self.prj[nt] = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_out, bias=False),
                nn.GELU())
            _init_linear(self.prj[nt][1])

    def forward(self, x, ts):
        out={}
        for nt,xn in x.items():
            if ts and ts[nt] is not None:
                xn = torch.cat([xn, self.t2v(ts[nt])], dim=1)
            out[nt]=self.prj[nt](xn)
        return out


class RGTLayer(nn.Module):
    def __init__(self, d=hidden_dim, h=8, p=0.1, time_decay=True, device=device):
        super().__init__()
        self.h = h
        self.dk = d // h
        self.d = d
        self.device = device
        self.Q = nn.Linear(self.d, self.d, bias=False)
        self.K = nn.Linear(self.d, self.d, bias=False)
        self.V = nn.Linear(self.d, self.d, bias=False)
        self.time_decay = time_decay
        self.rel = nn.ParameterDict()
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4*d, bias=False),
            nn.GELU(),
            nn.Linear(4*d, d, bias=False),
            nn.Dropout(p))

    def _r(self,r):
        k='__'.join(r)
        if k not in self.rel:
            param = nn.Parameter(torch.randn(self.h, self.dk, device=self.device))
            self.register_parameter(k, param)
            self.rel[k] = param
        else:
            param = self.rel[k]
            if param.device != self.device:
                param.data = param.data.to(self.device)
        return param

    def forward(self, h, ei, ew, ts=None):
        msg = {k: torch.zeros_like(v) for k, v in h.items()}
        for r, e in ei.items():
            s, t = e
            q = self.Q(h[r[2]])[t].view(-1, self.h, self.dk)
            k = self.K(h[r[0]])[s].view(-1, self.h, self.dk)
            v = self.V(h[r[0]])[s].view(-1, self.h, self.dk)
            R = self._r(r)
            l = ((q * (k + R)).sum(-1)) / np.sqrt(self.dk)
            if self.time_decay and ts is not None:
                dt = 1.0 - ts[s]
                l = l - dt.unsqueeze(-1)
            if ew and ew.get(r) is not None:
                l = l + ew[r].unsqueeze(-1)
                p = F.softmax(l, dim=-1)
                m = (p.unsqueeze(-1) * v).view(-1, self.d)
                msg[r[2]].index_add_(0, t, m)
        o = {}
        for nt in h:
            h_res = h[nt] + msg[nt]
            o[nt] = h_res + self.ffn(h_res)
        return o

class PolarityAwareConv(nn.Module):
    def __init__(self, h_dim, e_dim, p):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(e_dim - 1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p),
        )

    def forward(self, x, ei, ea):
        x_src, x_dst = x
        src, dst = ei

        pol = ea[:, :1].clamp(0, 1)
        raw = ea[:, 1:]
        if raw.dim() == 1:
            raw = raw.unsqueeze(-1)

        e_feat = self.edge_mlp(raw) * (pol + 0.01)
        src_emb = x_src[src]
        m = src_emb * e_feat

        return scatter_add(m, dst, dim=0, dim_size=x_dst.size(0))

class ActorHead(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h = h
        dk = d // h
        self.Q = nn.Linear(d, d, bias=False)
        self.K = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, d, bias=False)
        self.dk = dk

    def forward(self, a_z, bv_z, mask, weight=None):
        q = self.Q(a_z).view(-1, self.h, self.dk).transpose(0, 1)
        k = self.K(bv_z).view(-1, self.h, self.dk).transpose(0, 1)
        v = self.V(bv_z).view(-1, self.h, self.dk).transpose(0, 1)
        att = (q @ k.transpose(-1, -2)) / np.sqrt(self.dk)
        att = att.transpose(0, 1)
        att = att.masked_fill(~mask.unsqueeze(1), -1e9)
        if weight is not None:
            att = att + weight.unsqueeze(1)
        p = att.softmax(dim=-1)
        context = torch.matmul(p.transpose(0, 1), v)
        topic_align = context.mean(0)
        influence = p.mean(1).sum(-1)
        return topic_align, influence

class BillTopicHead(nn.Module):
    def __init__(self, hidden_dim, k):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, k, bias=False)
    def forward(self, z):
        return self.fc(z)

class SuccessHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l = nn.Linear(d, 1)

    def forward(self, z):
        return self.l(z).squeeze(-1)

class HierarchyAggregator(nn.Module):
    def forward(self, z, b):
        s, d = b.edge_index_dict[('bill_version', 'is_version', 'bill')]
        bill_agg = scatter_mean(
            z['bill_version'][s],
            d,
            dim=0,
            dim_size=z['bill'].size(0)
        )
        z['bill'] = 0.7 * z['bill'] + 0.3 * bill_agg
        d2, s2 = b.edge_index_dict[('legislator', 'samePerson', 'legislator_term')]
        leg_agg = scatter_mean(
            z['legislator_term'][s2],
            d2,
            dim=0,
            dim_size=z['legislator'].size(0)
        )
        z['legislator'] = 0.7 * z['legislator'] + 0.3 * leg_agg
        return z

class LegislativeGraphEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim, n_layers, dropout, device=device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]

        in_dims = {nt: c for nt, c in zip(self.node_types, [769, 389, 385, 2, 385, 384, 384])}
        self.proj = FeatureProjector(metadata, in_dims, d_out=hidden_dim, t_dim=12)

        self.rgt_layers = nn.ModuleList([RGTLayer(hidden_dim, 4, dropout) for _ in range(n_layers)])

        vote_e_dim = 385
        self.vote_conv = PolarityAwareConv(hidden_dim, vote_e_dim, dropout)

        self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim).to(device) for nt in self.node_types})
        self.drop = nn.Dropout(dropout)

    def forward(self, batch):
        ts_dict = {nt: getattr(batch[nt], 'timestamp', None) for nt in self.node_types}
        h = self.proj({nt: batch[nt].x.to(self.device) for nt in self.node_types}, ts_dict)

        edge_w = {et: getattr(batch[et], 'edge_weight', None) for et in batch.edge_types}
        for layer in self.rgt_layers:
            h = layer(h, batch.edge_index_dict, edge_w)

        vote_ei = batch.edge_index_dict[('legislator_term', 'voted_on', 'bill_version')]
        vote_attr = batch[('legislator_term', 'voted_on', 'bill_version')].edge_attr
        vote_msg = self.vote_conv(
            (h["legislator_term"], h["bill_version"]),
            vote_ei,
            vote_attr,
        )
        h["bill_version"] += vote_msg

        h = {nt: self.drop(F.relu(self.norm[nt](h[nt]))) for nt in h}
        return h

class LegislativeGraphModel(nn.Module):
    def __init__(self, metadata, cluster_id, topic_onehot, hidden_dim, dropout, device=device):
        super().__init__()
        self.device = device
        self.encoder = LegislativeGraphEncoder(metadata, hidden_dim, n_layers, dropout, device)
        self.hier = HierarchyAggregator()
        self.topic_head = BillTopicHead(hidden_dim, topic_onehot.size(1))
        self.success = SuccessHead(hidden_dim)

        self.actor_types = ['legislator', 'committee', 'donor', 'lobby_firm']
        self.actor_head = nn.ModuleDict({nt: ActorHead(hidden_dim) for nt in self.actor_types})
        k_topics = topic_onehot.size(1)
        dk = hidden_dim // self.actor_head['legislator'].h
        self.actor_topic_proj = nn.ModuleDict({
            nt: nn.Linear(dk, k_topics, bias=False) for nt in self.actor_types
        })

        self.register_buffer('cluster_id',  cluster_id)
        self.register_buffer('topic_onehot', topic_onehot)

    def forward(self, batch, mask_weight_dict):
        z = self.hier(self.encoder(batch), batch)
        bill_nodes = batch['bill'].node_id.to(self.device)
        bv_nodes = batch['bill_version'].node_id.to(self.device)

        bill_embed = z['bill'][bill_nodes]
        z_bv = z['bill_version'][bv_nodes]

        bill_logits = self.topic_head(bill_embed)
        bill_labels = self.cluster_id[bill_nodes]
        bill_prob = self.topic_onehot[bill_nodes]

        rows_g, cols_g = batch.edge_index_dict[('bill_version','is_version','bill')]
        bv_pos = {nid: i for i, nid in enumerate(bv_nodes.tolist())}
        bill_pos = {nid: i for i, nid in enumerate(bill_nodes.tolist())}

        mask = (torch.isin(rows_g, bv_nodes) & torch.isin(cols_g, bill_nodes))
        if mask.any():
            r_idx = torch.tensor([bv_pos[int(r)]   for r in rows_g[mask]],
                                device=self.device, dtype=torch.long)
            c_idx = torch.tensor([bill_pos[int(c)] for c in cols_g[mask]],
                                device=self.device, dtype=torch.long)

            W_vb = torch.zeros(len(bv_nodes), len(bill_nodes), device=self.device)
            W_vb[r_idx, c_idx] = 1.0
        else:
            W_vb = torch.zeros(len(bv_nodes), len(bill_nodes), device=self.device)
        z_actor = {nt: z[nt] for nt in self.actor_types if nt in z}
        success_log = self.success(bill_embed)
        succ_scalar = torch.sigmoid(success_log).mean()

        align_dict, infl_dict = {},{}
        for nt, az in z_actor.items():
            mvb, wvb = mask_weight_dict[nt]
            mvb, wvb = mvb.to(self.device), wvb.to(self.device)
            ta, inf = self.actor_head[nt](az, z_bv, mvb, wvb)
            logits = self.actor_topic_proj[nt](ta)
            align_dict[nt] = logits.softmax(-1)
            infl_dict[nt] = inf * succ_scalar

        return {
            "node_embeddings": z,
            "bill_logits": bill_logits,
            "bill_labels": bill_labels,
            "success_logit": success_log,
            "actor_topic_dict": align_dict,
            "influence_dict" : infl_dict
        }

    def compute_loss(self, out, batch, neg_k=1, temp=0.1):
        ce = F.cross_entropy(out["bill_logits"], out["bill_labels"])
        if torch.isnan(ce) or torch.isinf(ce):
            ce = torch.tensor(0.0, device=self.device)
        link = 0.0
        bce = nn.BCEWithLogitsLoss()
        z = out['node_embeddings']
        for (s_t, _, d_t), ei in batch.edge_index_dict.items():
            s, d = ei
            pos = (F.normalize(z[s_t][s], -1) * F.normalize(z[d_t][d], -1)).sum(-1) / temp
            link += bce(pos, torch.ones_like(pos))
            if neg_k:
                perm = d[torch.randperm(d.size(0), device=self.device)][: pos.size(0) * neg_k]
                neg = (
                    F.normalize(z[s_t][s].repeat(neg_k, 1), -1)
                    * F.normalize(z[d_t][perm], -1)
                ).sum(-1) / temp
                link += bce(neg, torch.zeros_like(neg))

        infl_vals = []
        for v in out['influence_dict'].values():
            v = v[torch.isfinite(v)]
            if v.numel() > 0:
                infl_vals.append(v.pow(2).mean())
        inf_reg = torch.stack(infl_vals).mean() if infl_vals else torch.tensor(0., device=self.device)

        ent_values = []
        for probs in out['actor_topic_dict'].values():
            p = torch.nan_to_num(probs, nan=0.0, posinf=1e8, neginf=-1e2)
            if p.sum() == 0:
                continue
            row_sum = p.sum(dim=1, keepdim=True)
            mask = (row_sum.squeeze(-1) > 0)
            p_norm = torch.zeros_like(p)
            p_norm[mask] = p[mask] / row_sum[mask]
            entropy = (-p_norm[mask] * (p_norm[mask] + 1e-9).log()).sum(dim=1)
            if entropy.numel() > 0:
                ent_values.append(entropy.mean())
        ent_reg = torch.stack(ent_values).mean() if ent_values else torch.tensor(0., device=self.device)

        total = ce + link + 0.1 * inf_reg + 0.1 * ent_reg
        return {
            'total_loss': total,
            'ce_loss': ce,
            'link_loss': link,
            'inf_reg': inf_reg,
            'ent_reg': ent_reg
        }
def _adj(data, et):
    row, col = data.edge_index_dict[et]
    ns, nd = data[et[0]].num_nodes, data[et[2]].num_nodes
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), torch.ones_like(row, dtype=torch.float32), (ns, nd)
    ).coalesce()

def build_masks_weights(d, acts, A_by, paths, device=device):
    dev = torch.device('cpu')
    rows = {nt: d[nt].node_id.cpu() for nt in acts}
    cols = d['bill_version'].node_id.cpu()
    out = {}
    col_map = {int(g): i for i, g in enumerate(cols.tolist())}
    for nt in acts:
        n_ids = rows[nt]
        row_map = {int(g): i for i, g in enumerate(n_ids.tolist())}
        n_r = n_ids.size(0)
        n_c = cols.size(0)
        m = torch.zeros(n_r, n_c, dtype=torch.bool, device=dev)
        w = torch.zeros(n_r, n_c, dtype=torch.float32, device=dev)

        for path in paths[nt]:
            S = A_by[path[0]]
            for et in path[1:]:
                S = (S @ A_by[et]).coalesce()

            r, c = S.indices()
            keep = torch.isin(r, rows[nt]) & torch.isin(c, cols)
            if not keep.any():
                continue
            r_kept = r[keep].tolist()
            c_kept = c[keep].tolist()
            r_sub = [row_map[int(g)] for g in r_kept]
            c_sub = [col_map[int(g)] for g in c_kept]
            m[r_sub, c_sub] = True
            if any(e in path[0][1] or e in path[1][1] for e in {'donated_to','wrote','lobbied','member_of'}):
                ei = A_by[path[0]].indices()
                ea = A_by[path[0]].values()
                Wsp = torch.sparse_coo_tensor(ei, ea, A_by[path[0]].shape, device=dev).coalesce()
                for et in path[1:]:
                    Wsp = (Wsp @ A_by[et]).coalesce()

                wr, wc = Wsp.indices()
                keep_w = torch.isin(wr, rows[nt]) & torch.isin(wc, cols)
                if keep_w.any():
                    wr_sub = [row_map[int(g)] for g in wr[keep_w].tolist()]
                    wc_sub = [col_map[int(g)] for g in wc[keep_w].tolist()]
                    w[wr_sub, wc_sub] += Wsp.values()[keep_w]

        out[nt] = (m.to(device), w.to(device))
    return out

def main():
    with open('bill_labels.json', 'r') as f:
        topic_cluster_labels_dict = json.load(f)

    data = load_and_preprocess_data()
    num_topics = len(list(set([v for v in topic_cluster_labels_dict.values()]))) + 1
    cluster_bill = torch.full((data['bill'].num_nodes,), num_topics, dtype=torch.long)
    key1 = data['bill'].n_id.tolist()
    key2 = data['bill'].node_id.tolist()
    key = {k1: k2 for k1, k2 in zip(key1, key2)}
    for bill_nid, lab in topic_cluster_labels_dict.items():
        if bill_nid in key:
            cluster_bill[key[bill_nid]] = lab

    topic_onehot_bill = F.one_hot(cluster_bill.clamp(max=num_topics-1), num_classes=num_topics).float()
    cluster_bill, topic_onehot_bill = cluster_bill.to(device), topic_onehot_bill.to(device)
    A_by = {et: _adj(data, et).coalesce() for et in data.edge_types}

    ALLOWED = {
        "donor": [
            ( ('donor','donated_to','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            (
            ('donor', 'donated_to','legislator_term'),
            ('legislator_term','wrote','bill_version') ),
        ],
        "lobby_firm": [
            ( ('lobby_firm','lobbied','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('lobby_firm','lobbied','committee'),
            ('committee','rev_member_of','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('lobby_firm','lobbied','committee'),
            ('committee','rev_member_of','legislator_term'),
            ('legislator_term','wrote','bill_version') ),
            ( ('lobby_firm','lobbied','legislator_term'),
            ('legislator_term', 'wrote','bill_version') )
        ],
        "committee": [
            ( ('committee','rev_member_of','legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('committee','rev_member_of','legislator_term'),
            ('legislator_term','wrote','bill_version') )
        ],
        "legislator": [
            ( ('legislator', 'samePerson', 'legislator_term'),
            ('legislator_term','voted_on','bill_version') ),
            ( ('legislator', 'samePerson', 'legislator_term'),
            ('legislator_term','wrote','bill_version') )
        ]
    }

    model = LegislativeGraphModel(data.metadata(), cluster_bill, topic_onehot_bill, hidden_dim, dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    torch.mps.empty_cache()
    gc.collect()
    for nt in data.node_types:
        del data[nt].n_id
    loader = HGTLoader(
        data,
        num_samples={
            'legislator_term': [64] * 2,
            'legislator': [64] * 2,
            'committee': [96] * 2,
            'lobby_firm': [86] * 2,
            'donor': [86] * 2,
            'bill_version': [832] * 2,
            'bill': [684] * 2,
        },
        batch_size=128,
        shuffle=True,
        input_nodes='legislator_term'
    )

    epochs = 25
    patience = 5
    counter = 0
    start_epoch = 0
    best_loss = float('inf')
    # model.load_state_dict(torch.load("best_model5.pt"), strict=False)
    # optimizer.load_state_dict(torch.load("best_opt5.pt"))

    for epoch in tqdm(range(start_epoch, start_epoch + epochs), position=0, total=epochs):
        model.train()
        epoch_loss = 0.0
        torch.mps.empty_cache()
        batch_count = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", position=1)):
            batch = batch.to(device)
            mw_dict = build_masks_weights(batch, ['legislator','committee','donor','lobby_firm'], A_by, ALLOWED)
            outputs = model(batch, mw_dict)
            loss_dict = model.compute_loss(outputs, batch)
            loss = loss_dict['total_loss'].float()

            if torch.isnan(loss) or torch.isinf(loss):
                print(loss_dict)
                alarm()
                raise ValueError(f"Batch {batch_idx}: NaN/Inf loss detected!")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            del outputs, loss_dict, loss
            torch.mps.empty_cache()
            gc.collect()
            batch_count += 1

        avg_loss = epoch_loss / max(1, batch_count)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "best_model5.pt")
            torch.save(optimizer.state_dict(), "best_opt5.pt")
            print(f"  New best loss; resetting patience counter.")
        else:
            counter += 1
            print(f"  No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    main()